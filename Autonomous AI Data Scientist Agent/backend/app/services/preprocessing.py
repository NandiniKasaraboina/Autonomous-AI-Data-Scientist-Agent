"""Automatic data cleaning.

Pipeline (each step is logged so the UI / PDF report can explain what happened):
  1. Rule based   : normalise text, hidden missing tokens, empty rows/cols, duplicates, types, dates,
                    inconsistent spelling, useless columns.
  2. Machine learning : ExtraTrees iterative imputer or KNN imputer fills missing values using the
                    relationships between columns (not just a mean).
  3. ML + Deep learning : IsolationForest and a denoising autoencoder vote on anomalous rows; the
                    autoencoder then repairs the broken cells (or rows are capped / removed).
"""
from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from ..utils import df_records, is_categorical, is_datetime, is_numeric
from .dl import TabularAutoencoder

NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", "#n/a", "nil", "#na", "<na>"}
IMPUTE_SAMPLE = 8000


def _int_like(s: pd.Series) -> bool:
    v = s.dropna()
    return len(v) > 0 and bool(np.all(np.isclose(v.astype(float) % 1, 0)))


def _step(steps: list, title: str, method: str, detail: str, items: Optional[list] = None, status: str = "done"):
    steps.append({"title": title, "method": method, "detail": detail, "items": items or [], "status": status})


# ----------------------------------------------------------------------------- rule based
def _clean_text_and_tokens(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    hidden = 0

    def fix(v):
        if isinstance(v, str):
            v = v.strip()
            return np.nan if v.lower() in NA_TOKENS else v
        return v

    for c in df.columns:
        if is_categorical(df[c]):
            before = df[c].isna().sum()
            df[c] = df[c].astype(object).map(fix)
            hidden += int(df[c].isna().sum() - before)
    return df, hidden


def _to_numeric_if_possible(s: pd.Series) -> Optional[pd.Series]:
    nonnull = s.dropna()
    if nonnull.empty:
        return None
    txt = s.astype(str).str.replace(r"[,$€£₹%\s]", "", regex=True)
    num = pd.to_numeric(txt, errors="coerce")
    if num[s.notna()].notna().mean() >= 0.9:
        return num
    return None


_DATE_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def _to_datetime_if_possible(s: pd.Series) -> Optional[pd.Series]:
    nonnull = s.dropna().astype(str)
    if nonnull.empty:
        return None
    sample = nonnull.head(200)
    if sample.map(lambda v: bool(_DATE_RE.match(v))).mean() < 0.8:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce")
    return parsed if parsed[s.notna()].notna().mean() >= 0.9 else None


def _standardise_spelling(s: pd.Series) -> tuple[pd.Series, int]:
    nn = s.dropna()
    if nn.empty or not all(isinstance(v, str) for v in nn.head(500)) or nn.nunique() > 200:
        return s, 0
    raw = nn.astype(str)
    key = raw.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    if raw.nunique() == key.nunique():
        return s, 0
    canon = pd.DataFrame({"raw": raw, "key": key}).groupby("key")["raw"].agg(lambda x: x.value_counts().idxmax())
    new = key.map(canon)
    changed = int((new != raw).sum())
    out = s.copy().astype(object)
    out.loc[nn.index] = new.values
    return out, changed


# ----------------------------------------------------------------------------- ML imputation
def _impute(df: pd.DataFrame, target: Optional[str], method: str, steps: list, int_cols: set) -> tuple[pd.DataFrame, dict]:
    missing_cols = [c for c in df.columns if df[c].isna().any()]
    info: dict = {"method": method, "columns": [], "cells_filled": 0}
    if not missing_cols:
        _step(steps, "Missing value imputation", "Machine learning", "No missing values left to fill.", status="skipped")
        return df, info

    num_cols = [c for c in df.columns if is_numeric(df[c])]
    cat_cols = [c for c in df.columns if is_categorical(df[c])]
    counts = {c: int(df[c].isna().sum()) for c in missing_cols}
    masks = {c: df[c].isna() for c in missing_cols}
    used = method
    if method in {"iterative", "knn"} and df.shape[1] < 2:
        used = "simple"

    out = df.copy()
    if used in {"simple", "mean", "median", "most_frequent"}:
        for c in missing_cols:
            if c in num_cols:
                if used == "mean":
                    fill = out[c].mean()
                elif used == "most_frequent":
                    fill = out[c].mode().iloc[0]
                else:
                    fill = out[c].median()
                out[c] = out[c].fillna(fill)
            else:
                out[c] = out[c].fillna(out[c].mode().iloc[0])
        label = {
            "mean": "Mean / most-frequent value",
            "median": "Median / most-frequent value",
            "most_frequent": "Most-frequent value",
            "simple": "Median / most-frequent value",
        }[used]
    else:
        M = pd.DataFrame(index=df.index)
        cats = {}
        for c in num_cols:
            M[c] = df[c].astype(float)
        for c in cat_cols:
            vals = df[c].astype(str).where(df[c].notna(), np.nan)
            cat = pd.Categorical(vals)
            cats[c] = cat.categories
            codes = cat.codes.astype(float)
            codes[codes < 0] = np.nan
            M[c] = codes
        arr = M.values
        rng = np.random.default_rng(42)
        fit_idx = np.arange(len(M)) if len(M) <= IMPUTE_SAMPLE else rng.choice(len(M), IMPUTE_SAMPLE, replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if used == "knn":
                scaler = StandardScaler().fit(arr[fit_idx])
                imp = KNNImputer(n_neighbors=5, weights="distance", keep_empty_features=True)
                imp.fit(scaler.transform(arr[fit_idx]))
                filled = scaler.inverse_transform(imp.transform(scaler.transform(arr)))
                label = "KNN imputer (5 nearest rows)"
            else:
                est = ExtraTreesRegressor(n_estimators=25, min_samples_leaf=2, n_jobs=-1, random_state=42)
                imp = IterativeImputer(estimator=est, max_iter=4, initial_strategy="median", random_state=42,
                                       skip_complete=True, keep_empty_features=True)
                imp.fit(arr[fit_idx])
                filled = imp.transform(arr)
                label = "Iterative imputer with ExtraTrees (each column predicted from the others)"
        F = pd.DataFrame(filled, columns=M.columns, index=M.index)
        for c in missing_cols:
            m = masks[c]
            if c in cats:
                codes = np.clip(np.rint(F.loc[m, c].values), 0, len(cats[c]) - 1).astype(int)
                out.loc[m, c] = np.asarray(cats[c])[codes]
            else:
                lo, hi = df[c].min(), df[c].max()
                out.loc[m, c] = np.clip(F.loc[m, c].values, lo, hi)
    for c in missing_cols:
        if c in int_cols and c in out and is_numeric(out[c]):
            out[c] = out[c].round()
        info["columns"].append({"column": c, "filled": counts[c], "kind": "numeric" if c in num_cols else "categorical"})
    info["cells_filled"] = int(sum(counts.values()))
    info["method"] = label
    top = sorted(info["columns"], key=lambda x: -x["filled"])[:8]
    _step(steps, "Missing value imputation", "Machine learning",
          f"Filled {info['cells_filled']:,} missing cells in {len(missing_cols)} column(s) using: {label}.",
          [f"{t['column']}: {t['filled']:,} filled" for t in top])
    return out, info


# ----------------------------------------------------------------------------- ML + DL outliers
def _outliers(df: pd.DataFrame, target: Optional[str], treatment: str, contamination: float,
              steps: list, int_cols: set, exclude: Optional[set] = None) -> tuple[pd.DataFrame, dict]:
    exclude = exclude or set()
    num_cols = [c for c in df.columns if is_numeric(df[c]) and c != target and c not in exclude and df[c].nunique() > 5]
    info: dict = {"ran": False, "treatment": treatment}
    if len(num_cols) < 2 or len(df) < 50:
        _step(steps, "Outlier detection and repair", "Machine learning + Deep learning",
              "Skipped: needs at least 2 continuous numeric columns and 50 rows.", status="skipped")
        return df, info

    X = df[num_cols].astype(float).values
    scaler = RobustScaler().fit(X)
    Xs = np.clip(scaler.transform(X), -10, 10)

    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1).fit(Xs)
    if_flag = iso.predict(Xs) == -1

    # Pass 1 learns roughly what normal rows look like; pass 2 retrains without the 10% worst-fitting
    # rows so the network cannot "memorise" the very outliers it should detect (trimmed training).
    ae = TabularAutoencoder().fit(Xs)
    first = np.abs(Xs - ae.reconstruct(Xs)).max(axis=1)
    ae = TabularAutoencoder().fit(Xs[first <= np.percentile(first, 90)])
    Xh = ae.reconstruct(Xs)
    err = np.log(np.abs(Xs - Xh).max(axis=1) ** 2 + 1e-9)   # worst cell of each row, log scale
    med = np.median(err)
    mad = 1.4826 * np.median(np.abs(err - med)) + 1e-9
    z = (err - med) / mad
    ae_flag = z > 2.5
    # third (statistical) voter: a value far outside the usual range of its column (3 x IQR fences)
    q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lo3, hi3 = q1 - 3 * iqr, q3 + 3 * iqr
    extreme_cells = (iqr[None, :] > 0) & ((X < lo3[None, :]) | (X > hi3[None, :]))
    extreme_flag = extreme_cells.any(axis=1)
    # "soft" Isolation Forest vote: the 10% most isolated rows
    if_soft = iso.score_samples(Xs) <= np.percentile(iso.score_samples(Xs), 10)
    # A row is confirmed when the models and the range check back each other up (or the autoencoder is very sure)
    flagged = (if_flag & (ae_flag | extreme_flag)) | (extreme_flag & if_soft) | (z > 5.0)

    info.update({
        "ran": True, "columns_used": num_cols, "autoencoder_backend": ae.backend,
        "loss_curve": ae.loss_curve(), "isolation_forest_flagged": int(if_flag.sum()),
        "autoencoder_flagged": int(ae_flag.sum()), "range_check_flagged": int(extreme_flag.sum()),
        "consensus_flagged": int(flagged.sum()),
        "cells_repaired": 0, "rows_removed": 0, "samples": [],
    })

    # 2-D projection for the UI scatter
    try:
        proj = PCA(n_components=2, random_state=42).fit_transform(Xs)
        idx = np.arange(len(Xs))
        if len(idx) > 500:
            keep = np.concatenate([np.where(flagged)[0][:100], np.random.default_rng(1).choice(idx, 400, replace=False)])
            idx = np.unique(keep)
        info["projection"] = [{"x": float(proj[i, 0]), "y": float(proj[i, 1]), "outlier": bool(flagged[i])} for i in idx]
    except Exception:
        info["projection"] = []

    out = df.copy()
    n_flag = int(flagged.sum())
    if n_flag == 0 or treatment == "none":
        detail = (f"Isolation Forest flagged {int(if_flag.sum())} rows, the autoencoder {int(ae_flag.sum())} and the range check "
                  f"{int(extreme_flag.sum())}; {n_flag} rows were confirmed as outliers by voting. " + ("No changes were made." if n_flag == 0 else "Treatment set to 'none', data left untouched."))
        _step(steps, "Outlier detection and repair", "Machine learning + Deep learning", detail,
              [f"Autoencoder backend: {ae.backend}"])
        return out, info

    if treatment == "remove":
        out = out.loc[~flagged].reset_index(drop=True)
        info["rows_removed"] = n_flag
        action = f"Removed {n_flag} anomalous rows."
    elif treatment == "cap":
        q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cells = 0
        for j, c in enumerate(num_cols):
            if iqr[j] == 0:
                continue
            col = out[c].astype(float).values.copy()
            m = flagged & ((col < lo[j]) | (col > hi[j]))
            for i in np.where(m)[0][:3]:
                if len(info["samples"]) < 10:
                    info["samples"].append({"row": int(df.index[i]), "column": c, "before": float(col[i]),
                                            "after": float(np.clip(col[i], lo[j], hi[j]))})
            col[m] = np.clip(col[m], lo[j], hi[j])
            cells += int(m.sum())
            out[c] = col
        info["cells_repaired"] = cells
        action = f"Capped {cells} extreme cells in {n_flag} anomalous rows to the IQR fences."
    else:  # reconstruct with the autoencoder
        resid = np.abs(Xs - Xh)
        r_med = np.median(resid, axis=0)
        r_mad = 1.4826 * np.median(np.abs(resid - r_med), axis=0)
        thr = np.maximum(r_med + 4 * r_mad, 0.75)
        cells_mask = flagged[:, None] & (((resid > thr[None, :]) & (np.abs(Xs) > 2.5)) | extreme_cells)
        # autoencoder reconstruction, kept inside the plausible (Tukey 1.5 x IQR) range of each column
        Xr = scaler.inverse_transform(Xh)
        lo15 = np.where(iqr > 0, np.maximum(X.min(axis=0), q1 - 1.5 * iqr), X.min(axis=0))
        hi15 = np.where(iqr > 0, np.minimum(X.max(axis=0), q3 + 1.5 * iqr), X.max(axis=0))
        Xr = np.clip(Xr, lo15[None, :], hi15[None, :])
        new = X.copy()
        new[cells_mask] = Xr[cells_mask]
        for j, c in enumerate(num_cols):
            col = new[:, j]
            if c in int_cols:
                col = np.rint(col)
            out[c] = col
        rows, cols = np.where(cells_mask)
        for i, j in list(zip(rows, cols))[:10]:
            info["samples"].append({"row": int(df.index[i]), "column": num_cols[j],
                                    "before": float(X[i, j]), "after": float(new[i, j])})
        info["cells_repaired"] = int(cells_mask.sum())
        action = f"Autoencoder repaired {info['cells_repaired']} corrupted cells in {n_flag} anomalous rows."

    _step(steps, "Outlier detection and repair", "Machine learning + Deep learning",
          f"Isolation Forest flagged {int(if_flag.sum())} rows, the autoencoder {int(ae_flag.sum())} and the range check "
          f"{int(extreme_flag.sum())}; {n_flag} rows were confirmed as outliers by voting. {action}",
          [f"Autoencoder backend: {ae.backend}", f"Columns analysed: {', '.join(num_cols[:8])}{'...' if len(num_cols) > 8 else ''}"])
    return out, info


# ----------------------------------------------------------------------------- main entry
def run_preprocessing(raw: pd.DataFrame, target: Optional[str] = None, imputation: str = "iterative",
                      outlier_treatment: str = "reconstruct", drop_duplicates: bool = True,
                      missing_col_threshold: float = 0.6, contamination: float = 0.03,
                      categorical_encoding: str = "onehot") -> tuple[pd.DataFrame, dict]:
    if target and target not in raw.columns:
        raise HTTPException(400, f"Target column '{target}' not found in the dataset.")
    steps: list = []
    df = raw.copy()
    rows0, cols0 = df.shape
    raw_missing = int(df.isna().sum().sum())

    # 1. text clean-up + hidden missing tokens
    df, hidden = _clean_text_and_tokens(df)
    missing_before = int(df.isna().sum().sum())
    _step(steps, "Text clean-up", "Rule-based",
          f"Trimmed whitespace and converted {hidden:,} hidden missing tokens (such as 'N/A', '?', 'null') to real missing values.")

    # 2. empty rows / columns
    empty_cols = [c for c in df.columns if df[c].isna().all() and c != target]
    df = df.drop(columns=empty_cols)
    n_before = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    _step(steps, "Remove empty rows and columns", "Rule-based",
          f"Removed {len(empty_cols)} empty column(s) and {n_before - len(df)} empty row(s).", empty_cols)

    # 3. duplicates
    dups = int(df.duplicated().sum())
    if drop_duplicates and dups:
        df = df.drop_duplicates().reset_index(drop=True)
    _step(steps, "Remove duplicate rows", "Rule-based",
          f"Removed {dups:,} duplicate rows." if drop_duplicates else f"Found {dups:,} duplicate rows (removal disabled).",
          status="done" if drop_duplicates else "skipped")

    # 4. type inference
    conversions = []
    for c in list(df.columns):
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            df[c] = s.astype(int)
            conversions.append(f"{c}: boolean -> 0/1")
        elif is_categorical(s):
            num = _to_numeric_if_possible(s)
            if num is not None:
                df[c] = num
                conversions.append(f"{c}: text -> number")
                continue
            dt = _to_datetime_if_possible(s)
            if dt is not None:
                df[c] = dt
                conversions.append(f"{c}: text -> date")
    derived: set = set()
    date_cols = [c for c in df.columns if is_datetime(df[c])]
    for c in date_cols:
        if c == target:
            continue
        d = df[c]
        df[f"{c}_year"], df[f"{c}_month"] = d.dt.year, d.dt.month
        df[f"{c}_day"], df[f"{c}_dayofweek"] = d.dt.day, d.dt.dayofweek
        df = df.drop(columns=[c])
        derived.update({f"{c}_year", f"{c}_month", f"{c}_day", f"{c}_dayofweek"})
        conversions.append(f"{c}: date -> year, month, day, dayofweek")
    _step(steps, "Fix data types", "Rule-based",
          f"Converted {len(conversions)} column(s) to a usable type.", conversions, "done" if conversions else "skipped")

    # 5. inconsistent spelling
    fixed = []
    for c in [c for c in df.columns if is_categorical(df[c])]:
        new, changed = _standardise_spelling(df[c])
        if changed:
            df[c] = new
            fixed.append(f"{c}: {changed} values harmonised")
    _step(steps, "Harmonise category spelling", "Rule-based",
          f"Merged inconsistent spellings (e.g. 'male', 'MALE ', 'Male') in {len(fixed)} column(s).", fixed,
          "done" if fixed else "skipped")

    # 6. drop useless columns
    dropped = []
    n = len(df)
    for c in list(df.columns):
        if c == target:
            continue
        s = df[c]
        miss = s.isna().mean()
        nun = s.nunique(dropna=True)
        reason = None
        if miss > missing_col_threshold:
            reason = f"{miss:.0%} missing"
        elif nun <= 1:
            reason = "constant value"
        elif is_categorical(s) and n > 20 and nun / max(s.notna().sum(), 1) > 0.9:
            reason = "identifier / free text (almost every value unique)"
        elif is_numeric(s) and n > 20 and re.search(r"(^|[_\s])id$|^id([_\s]|$)", c.lower()) and nun / max(s.notna().sum(), 1) > 0.95:
            reason = "identifier column"
        if reason:
            dropped.append({"column": c, "reason": reason})
    df = df.drop(columns=[d["column"] for d in dropped])
    _step(steps, "Drop uninformative columns", "Rule-based",
          f"Dropped {len(dropped)} column(s) that cannot help analysis.", [f"{d['column']} ({d['reason']})" for d in dropped],
          "done" if dropped else "skipped")

    # 7. missing target
    if target:
        miss_t = int(df[target].isna().sum())
        if miss_t:
            df = df.dropna(subset=[target]).reset_index(drop=True)
        _step(steps, "Rows without a target value", "Rule-based",
              f"Removed {miss_t} row(s) where the target '{target}' is missing (they cannot be used for learning).",
              status="done" if miss_t else "skipped")

    if df.empty or df.shape[1] == 0:
        raise HTTPException(400, "Nothing is left after cleaning. Check the dataset and thresholds.")

    int_cols = {c for c in df.columns if is_numeric(df[c]) and _int_like(df[c])}

    # 8. ML imputation
    df, imp_info = _impute(df, target, imputation, steps, int_cols)
    # 9. ML + DL outliers
    df, out_info = _outliers(df, target, outlier_treatment, contamination, steps, int_cols, derived)

    df = df.reset_index(drop=True)
    # safety net: nothing should stay missing (except never for target)
    leftover = int(df.isna().sum().sum())
    if leftover:
        for c in df.columns:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median() if is_numeric(df[c]) else df[c].mode().iloc[0])

    total_cells_0 = max(rows0 * cols0, 1)
    total_cells_1 = max(df.shape[0] * df.shape[1], 1)
    report = {
        "summary": {
            "rows_before": rows0, "rows_after": len(df), "cols_before": cols0, "cols_after": df.shape[1],
            "missing_before": missing_before, "missing_after": int(df.isna().sum().sum()),
            "raw_missing": raw_missing, "hidden_missing": missing_before - raw_missing,
            "completeness_before": round(100 * (1 - missing_before / total_cells_0), 2),
            "completeness_after": round(100 * (1 - int(df.isna().sum().sum()) / total_cells_1), 2),
            "duplicates_removed": dups if drop_duplicates else 0,
            "cells_imputed": imp_info.get("cells_filled", 0),
            "cells_repaired": out_info.get("cells_repaired", 0),
            "rows_removed_outliers": out_info.get("rows_removed", 0),
        },
        "options": {"target": target, "imputation": imputation, "outlier_treatment": outlier_treatment,
                    "drop_duplicates": drop_duplicates, "missing_col_threshold": missing_col_threshold,
                    "contamination": contamination, "categorical_encoding": categorical_encoding},
        "steps": steps, "imputation": imp_info, "outliers": out_info, "dropped_columns": dropped,
        "columns_after": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "preview_before": df_records(raw, 8), "preview_after": df_records(df, 8),
    }
    return df, report
