"""Exploratory data analysis: statistics + chart-ready data for the frontend and the PDF report."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from ..utils import df_records, is_categorical, is_datetime, is_numeric
from .ml import detect_task


def _hist(s: pd.Series, bins: int = 20) -> list[dict]:
    v = s.dropna().astype(float)
    if v.empty:
        return []
    counts, edges = np.histogram(v, bins=min(bins, max(v.nunique(), 1)))
    return [{"range": f"{edges[i]:.4g}", "label": f"{edges[i]:.4g} – {edges[i + 1]:.4g}", "count": int(counts[i])}
            for i in range(len(counts))]


def run_eda(df: pd.DataFrame, target: Optional[str] = None, source: str = "cleaned") -> dict:
    n_rows, n_cols = df.shape
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    cat_cols = [c for c in df.columns if is_categorical(df[c])]
    dt_cols = [c for c in df.columns if is_datetime(df[c])]

    columns, insights = [], []
    for c in df.columns:
        s = df[c]
        info = {"name": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "missing_pct": round(100 * s.isna().mean(), 2), "unique": int(s.nunique(dropna=True))}
        if c in num_cols:
            v = s.dropna().astype(float)
            q1, q3 = (v.quantile(.25), v.quantile(.75)) if len(v) else (np.nan, np.nan)
            iqr = q3 - q1
            n_out = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()) if iqr and iqr > 0 else 0
            info.update({"kind": "numeric", "mean": v.mean(), "std": v.std(), "min": v.min(), "q1": q1,
                         "median": v.median(), "q3": q3, "max": v.max(),
                         "skew": v.skew() if len(v) > 2 else None, "outliers_iqr": n_out})
            if info["skew"] is not None and abs(info["skew"]) > 1.5:
                insights.append({"level": "info", "text": f"'{c}' is strongly skewed (skew {info['skew']:.1f}); a log transform may help linear models."})
            if n_out > 0.05 * n_rows:
                insights.append({"level": "warning", "text": f"'{c}' has {n_out} outliers ({100 * n_out / n_rows:.1f}% of rows)."})
        elif c in dt_cols:
            info.update({"kind": "datetime", "min": str(s.min()), "max": str(s.max())})
        else:
            vc = s.astype(object).value_counts(dropna=True).head(6)
            info.update({"kind": "categorical", "top_values": [{"value": str(k), "count": int(v)} for k, v in vc.items()]})
        if info["missing_pct"] >= 20:
            insights.append({"level": "warning", "text": f"'{c}' is {info['missing_pct']:.0f}% missing."})
        columns.append(info)

    # histograms / categorical counts
    histograms = {c: _hist(df[c]) for c in num_cols[:12]}
    categorical = {}
    for c in cat_cols[:8]:
        vc = df[c].astype(object).value_counts(dropna=True).head(10)
        categorical[c] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]

    # correlations
    corr_cols = [c for c in num_cols if df[c].nunique() > 1][:15]
    correlation = {"columns": corr_cols, "matrix": []}
    pairs = []
    if len(corr_cols) >= 2:
        cm = df[corr_cols].corr(numeric_only=True)
        correlation["matrix"] = cm.round(3).values.tolist()
        for i in range(len(corr_cols)):
            for j in range(i + 1, len(corr_cols)):
                r = cm.iloc[i, j]
                if pd.notna(r):
                    pairs.append({"a": corr_cols[i], "b": corr_cols[j], "r": round(float(r), 3)})
        pairs.sort(key=lambda p: -abs(p["r"]))
        for p in pairs[:3]:
            if abs(p["r"]) >= 0.7:
                insights.append({"level": "info", "text": f"'{p['a']}' and '{p['b']}' are strongly correlated (r = {p['r']:.2f})."})

    scatter = []
    for p in pairs[:3]:
        samp = df[[p["a"], p["b"]]].dropna()
        samp = samp.sample(min(len(samp), 300), random_state=1)
        scatter.append({"x": p["a"], "y": p["b"], "r": p["r"],
                        "points": [{"x": float(a), "y": float(b)} for a, b in zip(samp[p["a"]], samp[p["b"]])]})

    # target analysis (mutual information ranks both numeric and categorical features)
    target_info = None
    if target and target in df.columns:
        task = detect_task(df[target])
        feats = [c for c in df.columns if c != target and not is_datetime(df[c])]
        sub = df.dropna(subset=[target])
        if len(sub) > 20000:
            sub = sub.sample(20000, random_state=1)
        mi_rank = []
        if feats and len(sub) > 10:
            Xm = pd.DataFrame(index=sub.index)
            for c in feats:
                col = sub[c]
                Xm[c] = col.fillna(col.median()) if is_numeric(col) else pd.factorize(col.astype(str))[0]
            disc = np.array([not is_numeric(sub[c]) for c in feats])
            y = sub[target]
            try:
                if task == "classification":
                    mi = mutual_info_classif(Xm.values, pd.factorize(y.astype(str))[0], discrete_features=disc, random_state=0)
                else:
                    mi = mutual_info_regression(Xm.values, y.astype(float).values, discrete_features=disc, random_state=0)
                mi_rank = sorted([{"feature": f, "score": round(float(m), 4)} for f, m in zip(feats, mi)], key=lambda d: -d["score"])[:15]
            except Exception:
                mi_rank = []
        dist = (df[target].astype(object).value_counts(dropna=True).head(12).items() if task == "classification" else None)
        target_info = {"name": target, "task": task,
                       "distribution": [{"value": str(k), "count": int(v)} for k, v in dist] if dist is not None else _hist(df[target]),
                       "mutual_information": mi_rank}
        if task == "classification":
            vc = df[target].value_counts(normalize=True)
            if len(vc) > 1 and vc.iloc[0] > 0.8:
                insights.append({"level": "warning", "text": f"Target '{target}' is imbalanced: '{vc.index[0]}' makes up {vc.iloc[0]:.0%} of rows."})

    dup = int(df.duplicated().sum())
    if dup:
        insights.append({"level": "warning", "text": f"{dup} duplicate rows are present."})
    if not insights:
        insights.append({"level": "good", "text": "No major data quality problems detected."})

    missing = sorted([{"column": c["name"], "missing": c["missing"], "pct": c["missing_pct"]} for c in columns if c["missing"] > 0],
                     key=lambda d: -d["missing"])
    return {
        "source": source,
        "overview": {"rows": n_rows, "columns": n_cols, "numeric": len(num_cols), "categorical": len(cat_cols),
                     "datetime": len(dt_cols), "missing_cells": int(df.isna().sum().sum()), "duplicates": dup,
                     "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1)},
        "columns": columns, "histograms": histograms, "categorical": categorical, "correlation": correlation,
        "top_correlations": pairs[:8], "scatter": scatter, "missing": missing, "target": target_info,
        "insights": insights, "preview": df_records(df, 10),
    }
