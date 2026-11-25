"""Random Forest training, evaluation, explainability helpers."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, precision_score, r2_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..utils import is_categorical, is_datetime, is_numeric


def detect_task(y: pd.Series) -> str:
    y = y.dropna()
    if not is_numeric(y):
        return "classification"
    nun = y.nunique()
    integer_like = bool(np.all(np.isclose(y.astype(float) % 1, 0)))
    if nun <= 2 or (integer_like and nun <= 15):
        return "classification"
    return "regression"


class TopKCategories(BaseEstimator, TransformerMixin):
    """Keeps the k most frequent categories of every column, groups the rest as '__other__', NaN as '__missing__'."""

    def __init__(self, k: int = 15):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.keep_ = {c: set(X[c].astype(str).value_counts().head(self.k).index) for c in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        for c in self.columns_:
            col = X[c].astype(str).where(X[c].notna(), "__missing__")
            out[c] = col.where(col.isin(self.keep_[c] | {"__missing__"}), "__other__")
        return out


class CategoryCodeEncoder(BaseEstimator, TransformerMixin):
    """Encodes categories as integer codes, preserving appearance or sorted order."""

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.categories_ = {}
        self.mapping_ = {}
        for column in self.columns_:
            values = frame[column].dropna().astype(str).unique().tolist()
            categories = sorted(values) if self.sorted_order else values
            self.categories_[column] = categories
            self.mapping_[column] = {value: index for index, value in enumerate(categories)}
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.columns_)
        encoded = np.empty((len(frame), len(self.columns_)), dtype=float)
        for index, column in enumerate(self.columns_):
            values = frame[column].astype("string").map(self.mapping_[column]).fillna(-1)
            encoded[:, index] = values.to_numpy(dtype=float)
        return encoded


class LabelCodeEncoder(CategoryCodeEncoder):
    sorted_order = False


class OrdinalCodeEncoder(CategoryCodeEncoder):
    sorted_order = True


CLASSIFICATION_MODELS = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "gradient_boosting": "Gradient Boosting",
    "decision_tree": "Decision Tree",
}

REGRESSION_MODELS = {
    "random_forest": "Random Forest",
    "linear_regression": "Linear Regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "gradient_boosting": "Gradient Boosting",
    "decision_tree": "Decision Tree",
}


def build_pipeline(num_cols, cat_cols, task, params=None, categorical_encoding="onehot",
                   model_name="random_forest") -> Pipeline:
    transformers = []
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if model_name in {"logistic_regression", "svm", "knn", "ridge", "lasso", "linear_regression"}:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        if categorical_encoding == "onehot":
            transformers.append(("cat_onehot", Pipeline([("topk", TopKCategories(15)),
                                                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))
        elif categorical_encoding == "label":
            transformers.append(("cat_label", LabelCodeEncoder(), cat_cols))
        elif categorical_encoding == "ordinal":
            transformers.append(("cat_ordinal", OrdinalCodeEncoder(), cat_cols))
        else:
            raise HTTPException(400, f"Unsupported categorical encoding '{categorical_encoding}'.")
    prep = ColumnTransformer(transformers, remainder="drop")
    p = dict(params or {})
    if task == "classification":
        if model_name == "random_forest":
            model = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=42, **p)
        elif model_name == "logistic_regression":
            model = LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42, **p)
        elif model_name == "svm":
            model = SVC(probability=True, class_weight="balanced", random_state=42, **p)
        elif model_name == "knn":
            model = KNeighborsClassifier(**p)
        elif model_name == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=42, **p)
        elif model_name == "decision_tree":
            model = DecisionTreeClassifier(class_weight="balanced", random_state=42, **p)
        else:
            raise HTTPException(400, f"Unsupported classification model '{model_name}'.")
    else:
        if model_name == "random_forest":
            model = RandomForestRegressor(n_jobs=-1, random_state=42, **p)
        elif model_name == "linear_regression":
            model = LinearRegression(**p)
        elif model_name == "ridge":
            model = Ridge(**p)
        elif model_name == "lasso":
            model = Lasso(max_iter=5000, **p)
        elif model_name == "svm":
            model = SVR(**p)
        elif model_name == "knn":
            model = KNeighborsRegressor(**p)
        elif model_name == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=42, **p)
        elif model_name == "decision_tree":
            model = DecisionTreeRegressor(random_state=42, **p)
        else:
            raise HTTPException(400, f"Unsupported regression model '{model_name}'.")
    return Pipeline([("prep", prep), ("model", model)])


def _feature_to_original(pipe: Pipeline, num_cols, cat_cols, categorical_encoding="onehot") -> list[str]:
    names = list(num_cols)
    if cat_cols and categorical_encoding == "onehot":
        enc = pipe.named_steps["prep"].named_transformers_["cat_onehot"].named_steps["oh"]
        for col, cats in zip(cat_cols, enc.categories_):
            names += [col] * len(cats)
    elif cat_cols:
        names += list(cat_cols)
    return names


def _downsample(xs, ys, n=60):
    if len(xs) <= n:
        return [{"x": float(a), "y": float(b)} for a, b in zip(xs, ys)]
    idx = np.linspace(0, len(xs) - 1, n).astype(int)
    return [{"x": float(xs[i]), "y": float(ys[i])} for i in idx]


def train_model(df: pd.DataFrame, target: str, test_size: float = 0.2, tune: bool = False,
                n_estimators: int = 300, categorical_encoding: str = "onehot",
                model_name: str = "random_forest") -> tuple[dict, dict, pd.DataFrame]:
    if target not in df.columns:
        raise HTTPException(400, f"Target column '{target}' not found.")
    data = df.dropna(subset=[target]).copy()
    if len(data) < 30:
        raise HTTPException(400, "At least 30 rows with a target value are needed to train a model.")
    task = detect_task(data[target])
    allowed = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    if model_name not in allowed:
        raise HTTPException(400, f"Model '{model_name}' is not available for {task}.")

    feats = [c for c in data.columns if c != target and not is_datetime(data[c]) and not data[c].isna().all()]
    if not feats:
        raise HTTPException(400, "No usable feature columns found.")
    num_cols = [c for c in feats if is_numeric(data[c])]
    cat_cols = [c for c in feats if is_categorical(data[c])]
    if categorical_encoding not in {"onehot", "label", "ordinal"}:
        raise HTTPException(400, f"Unsupported categorical encoding '{categorical_encoding}'.")
    X = data[feats]

    le = None
    classes = None
    if task == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(data[target].astype(str)), index=data.index)
        classes = [str(c) for c in le.classes_]
        if len(classes) > 30:
            raise HTTPException(400, f"Target '{target}' has {len(classes)} classes - that looks like a regression target or an ID column.")
        if len(classes) < 2:
            raise HTTPException(400, "The target has only one class; nothing to learn.")
    else:
        y = data[target].astype(float)

    stratify = y if (task == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

    model_params = {"n_estimators": n_estimators} if model_name in {"random_forest", "gradient_boosting"} else {}
    pipe = build_pipeline(num_cols, cat_cols, task, model_params, categorical_encoding, model_name)
    scoring = "f1_weighted" if task == "classification" else "r2"
    tuned_params = None
    if tune:
        if model_name == "random_forest":
            grid = {"model__n_estimators": [200, 300, 500], "model__max_depth": [None, 8, 12, 20],
                    "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", "log2", 0.5]}
        elif model_name in {"logistic_regression", "ridge", "lasso"}:
            grid = {"model__C": [0.01, 0.1, 1, 10]} if model_name == "logistic_regression" else (
                {"model__alpha": [0.01, 0.1, 1, 10]} if model_name in {"ridge", "lasso"} else {})
        elif model_name == "svm":
            grid = {"model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"]}
        elif model_name == "knn":
            grid = {"model__n_neighbors": [3, 5, 7, 11], "model__weights": ["uniform", "distance"]}
        elif model_name == "gradient_boosting":
            grid = {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.03, 0.1, 0.2],
                    "model__max_depth": [2, 3, 4]}
        else:
            grid = {"model__max_depth": [None, 4, 8, 12], "model__min_samples_leaf": [1, 2, 4]}
        if grid:
            search = RandomizedSearchCV(pipe, grid, n_iter=min(10, len(grid)), cv=3, scoring=scoring,
                                        random_state=42, n_jobs=1)
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
            tuned_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    else:
        pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    train_pred = pipe.predict(X_train)
    fitted_model = pipe.named_steps["model"]
    model_params = {}
    if hasattr(fitted_model, "n_estimators"):
        model_params["n_estimators"] = int(fitted_model.n_estimators)
    if hasattr(fitted_model, "get_params"):
        for key in ("C", "alpha", "n_neighbors", "max_depth", "learning_rate"):
            if key in fitted_model.get_params() and fitted_model.get_params()[key] is not None:
                model_params[key] = fitted_model.get_params()[key]
    results: dict = {"task": task, "target": target, "model": model_name,
                     "model_name": allowed[model_name], "features": feats, "n_train": len(X_train), "n_test": len(X_test),
                     "tuned": bool(tune), "best_params": tuned_params, "classes": classes,
                     "categorical_encoding": categorical_encoding,
                     "model_params": model_params}

    # cross-validation on the training split
    try:
        if task == "classification":
            k = int(min(5, y_train.value_counts().min()))
            cv = StratifiedKFold(k, shuffle=True, random_state=42) if k >= 2 else KFold(3, shuffle=True, random_state=42)
        else:
            cv = KFold(5, shuffle=True, random_state=42)
        cvs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        results["cv"] = {"metric": scoring, "mean": float(cvs.mean()), "std": float(cvs.std()), "folds": len(cvs)}
    except Exception:
        results["cv"] = None

    if task == "classification":
        proba = pipe.predict_proba(X_test)
        m = {"accuracy": accuracy_score(y_test, pred),
             "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
             "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
             "f1": f1_score(y_test, pred, average="weighted", zero_division=0)}
        try:
            if len(classes) == 2:
                m["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
                fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                results["roc_curve"] = _downsample(fpr, tpr, 60)
            elif len(np.unique(y_test)) == len(classes):
                m["roc_auc"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
        except Exception:
            pass
        labels = list(range(len(classes)))
        results["confusion_matrix"] = {"labels": classes, "matrix": confusion_matrix(y_test, pred, labels=labels).tolist()}
        p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=labels, zero_division=0)
        results["per_class"] = [{"class": classes[i], "precision": float(p[i]), "recall": float(r[i]),
                                 "f1": float(f[i]), "support": int(s[i])} for i in labels]
        base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        m["baseline_accuracy"] = accuracy_score(y_test, base.predict(X_test))
        m["train_accuracy"] = accuracy_score(y_train, train_pred)
        conf = proba.max(axis=1)
        pred_labels = [classes[int(i)] for i in pred]
        actual_labels = [classes[int(i)] for i in y_test]
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        m = {"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred), "rmse": rmse,
             "train_r2": r2_score(y_train, train_pred)}
        base = DummyRegressor().fit(X_train, y_train)
        m["baseline_r2"] = r2_score(y_test, base.predict(X_test))
        idx = np.random.default_rng(0).choice(len(y_test), min(200, len(y_test)), replace=False)
        results["actual_vs_predicted"] = [{"x": float(y_test.values[i]), "y": float(pred[i])} for i in idx]
        conf = None
        pred_labels = [float(v) for v in pred]
        actual_labels = [float(v) for v in y_test]
    results["metrics"] = {k: float(v) for k, v in m.items()}

    # feature importance: impurity (aggregated per original column) + permutation on the test split
    owner = _feature_to_original(pipe, num_cols, cat_cols, categorical_encoding)
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        importance_values = coef
    else:
        importance_values = np.ones(len(owner))
    agg = pd.Series(importance_values, index=owner).groupby(level=0).sum()
    agg = (agg / agg.sum() if agg.sum() else agg).sort_values(ascending=False)
    results["feature_importance"] = [{"feature": f, "importance": float(v)} for f, v in agg.items()][:20]
    try:
        Xp, yp = X_test, y_test
        if len(Xp) > 1500:
            sel = np.random.default_rng(0).choice(len(Xp), 1500, replace=False)
            Xp, yp = Xp.iloc[sel], yp.iloc[sel]
        pi = permutation_importance(pipe, Xp, yp, n_repeats=5, random_state=42, n_jobs=1, scoring=scoring)
        perm = sorted([{"feature": f, "importance": float(a), "std": float(b)}
                       for f, a, b in zip(feats, pi.importances_mean, pi.importances_std)], key=lambda d: -d["importance"])
        results["permutation_importance"] = perm[:20]
    except Exception:
        results["permutation_importance"] = []

    # prediction table (test split) + input schema for the "try a prediction" form
    pred_df = X_test.copy()
    pred_df.insert(0, "actual", actual_labels)
    pred_df.insert(1, "predicted", pred_labels)
    if conf is not None:
        pred_df.insert(2, "confidence", np.round(conf, 4))
    results["test_index"] = [int(i) for i in X_test.index]
    sample = pred_df.head(25).reset_index().rename(columns={"index": "row"})
    results["predictions_sample"] = sample[["row", "actual", "predicted"] + (["confidence"] if conf is not None else [])].to_dict("records")
    schema, background = [], {}
    for c in feats:
        if c in num_cols:
            v = data[c].dropna()
            schema.append({"name": c, "type": "number", "min": float(v.min()), "max": float(v.max()), "default": float(v.median())})
            background[c] = float(v.median())
        else:
            vc = data[c].astype(str).value_counts()
            schema.append({"name": c, "type": "category", "options": list(vc.head(30).index), "default": str(vc.index[0])})
            background[c] = str(vc.index[0])
    results["schema"] = schema

    bundle = {"pipeline": pipe, "features": feats, "num_cols": num_cols, "cat_cols": cat_cols, "target": target,
              "task": task, "classes": classes, "background": background, "X_test": X_test, "y_test": y_test,
              "X_train": X_train}
    return bundle, results, pred_df


def predict_one(bundle: dict, row: dict) -> dict:
    feats = bundle["features"]
    rec = {}
    for c in feats:
        v = row.get(c, None)
        if c in bundle["num_cols"]:
            try:
                rec[c] = float(v) if v not in (None, "") else np.nan
            except (TypeError, ValueError):
                raise HTTPException(400, f"'{c}' must be a number.")
        else:
            rec[c] = None if v in (None, "") else str(v)
    X = pd.DataFrame([rec], columns=feats)
    pipe = bundle["pipeline"]
    if bundle["task"] == "classification":
        proba = pipe.predict_proba(X)[0]
        order = np.argsort(-proba)
        return {"task": "classification", "prediction": bundle["classes"][int(order[0])],
                "probabilities": [{"class": bundle["classes"][int(i)], "probability": float(proba[i])} for i in order[:6]]}
    return {"task": "regression", "prediction": float(pipe.predict(X)[0])}
