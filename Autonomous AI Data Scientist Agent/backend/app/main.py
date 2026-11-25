"""ML Insight Studio - FastAPI backend.

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import io
import os
import re
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth, store
from .services import insights as ins_svc
from .services.eda import run_eda
from .services.loader import read_upload
from .services.ml import detect_task, predict_one, train_model
from .services.preprocessing import run_preprocessing
from .services.report import build_pdf
from .utils import df_records, is_categorical, is_datetime, is_numeric, safe_name, to_jsonable

MAX_UPLOAD_MB = 50
SESSION_COOKIE = "mlis_session"
SESSION_COOKIE_SECURE = os.environ.get(
    "MLAPP_COOKIE_SECURE",
    "true" if os.environ.get("MLAPP_ENV", "").lower() == "production" else "false",
).lower() in {"1", "true", "yes"}
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "MLAPP_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]

app = FastAPI(title="ML Insight Studio API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def check_request_origin(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse(status_code=403, content={"detail": "Request origin is not allowed."})
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"Unexpected server error: {exc}"})


# ----------------------------------------------------------------------------- schemas
class PreprocessRequest(BaseModel):
    target: Optional[str] = None
    imputation: Literal["iterative", "knn", "mean", "median", "most_frequent", "simple"] = "iterative"
    outlier_treatment: Literal["reconstruct", "cap", "remove", "none"] = "reconstruct"
    categorical_encoding: Literal["onehot", "label", "ordinal"] = "onehot"
    drop_duplicates: bool = True
    missing_col_threshold: float = Field(0.6, ge=0.1, le=1.0)
    contamination: float = Field(0.03, ge=0.005, le=0.2)


class TrainRequest(BaseModel):
    target: str
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    tune: bool = False
    n_estimators: int = Field(300, ge=50, le=1000)
    model: str = "random_forest"


class PredictRequest(BaseModel):
    values: dict


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class ExplainRequest(BaseModel):
    row: int


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=12, max_length=128)
    legacy_dataset_id: Optional[str] = None


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=12, max_length=128)


# ----------------------------------------------------------------------------- helpers
def _normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized):
        raise HTTPException(422, "Enter a valid email address.")
    return normalized


def _session_user(request: Request) -> dict:
    token = request.cookies.get(SESSION_COOKIE)
    user = auth.get_session_user(token) if token else None
    if not user:
        raise HTTPException(401, "Please sign in to continue.")
    return user


def _owned_dataset_user(ds_id: str, user: dict = Depends(_session_user)) -> dict:
    if not auth.owns_dataset(ds_id, user["id"]):
        raise HTTPException(404, "Dataset not found.")
    return user


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=auth.SESSION_SECONDS,
        httponly=True,
        secure=bool(SESSION_COOKIE_SECURE),
        samesite="lax",
        path="/",
    )


def _meta(ds_id: str) -> dict:
    return store.load_json(ds_id, "meta")


def _save_meta(ds_id: str, meta: dict) -> None:
    store.save_json(ds_id, "meta", meta)


def _status(ds_id: str) -> dict:
    return {"cleaned": store.has(ds_id, "clean.pkl"), "trained": store.has(ds_id, "model.joblib"),
            "report": store.has(ds_id, "report.pdf")}


def _working_df(ds_id: str, source: str = "auto") -> tuple[pd.DataFrame, str]:
    if source in ("auto", "cleaned") and store.has(ds_id, "clean.pkl"):
        return store.load_df(ds_id, "clean"), "cleaned"
    if source == "cleaned":
        raise HTTPException(404, "No cleaned dataset yet. Run Data Preprocessing first.")
    return store.load_df(ds_id, "raw"), "raw"


def _profile(df: pd.DataFrame) -> list[dict]:
    out = []
    for c in df.columns:
        s = df[c]
        values = s.dropna()
        kind = "numeric" if is_numeric(s) else "categorical" if is_categorical(s) else "datetime"
        prediction_task = None
        prediction_reason = None
        if values.empty:
            prediction_reason = "This column has no non-missing values."
        elif is_datetime(s):
            prediction_reason = "Date/time columns cannot be prediction targets."
        else:
            prediction_task = detect_task(values)
            if len(values) < 30:
                prediction_reason = "At least 30 non-missing rows are needed."
            elif prediction_task == "classification" and values.nunique() < 2:
                prediction_reason = "Classification needs at least 2 classes."
            elif prediction_task == "classification" and values.nunique() > 30:
                prediction_reason = "Classification targets can have at most 30 classes."
        out.append({"name": c, "dtype": str(s.dtype), "kind": kind,
                    "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True)),
                    "sample": [str(v) for v in values.head(3).tolist()],
                    "prediction_task": prediction_task,
                    "prediction_supported": prediction_reason is None,
                    "prediction_reason": prediction_reason})
    return out


def _invalidate(ds_id: str, *names: str) -> None:
    for n in names:
        p = store.folder(ds_id) / n
        if p.exists():
            p.unlink()


def _insights_payload(ds_id: str) -> dict:
    meta = _meta(ds_id)
    prep = store.load_json(ds_id, "preprocess", required=False)
    ml = store.load_json(ds_id, "ml", required=False)
    df, source = _working_df(ds_id)
    target = (ml or {}).get("target") or meta.get("target")
    eda = run_eda(df, target, source)
    result = ins_svc.build_findings(meta, prep, eda, ml)
    pdp = []
    if ml and store.has(ds_id, "model.joblib"):
        bundle = store.load_model(ds_id)
        order = [p["feature"] for p in ml.get("permutation_importance", []) if p["importance"] > 0] or [f["feature"] for f in ml["feature_importance"]]
        pdp = ins_svc.pdp_curves(bundle, order[:3])
    result["pdp"] = pdp
    result["suggested_questions"] = ins_svc.SUGGESTED
    result["test_rows"] = [r["row"] for r in (ml or {}).get("predictions_sample", [])]
    store.save_json(ds_id, "insights", result)
    store.save_json(ds_id, "eda", eda)
    return result


# ----------------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    from .services.dl import TORCH_OK
    return {"status": "ok", "deep_learning_backend": "pytorch" if TORCH_OK else "sklearn-mlp"}


@app.post("/api/auth/register", status_code=201)
def register(req: RegisterRequest, response: Response):
    name = req.name.strip()
    if len(name) < 2:
        raise HTTPException(422, "Name must contain at least 2 characters.")
    email = _normalize_email(req.email)
    try:
        user, token, adopted_dataset_id = auth.create_user(
            name, email, req.password, req.legacy_dataset_id
        )
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists.") from None
    _set_session_cookie(response, token)
    return {"user": user, "adopted_dataset_id": adopted_dataset_id}


@app.post("/api/auth/login")
def login(req: LoginRequest, request: Request, response: Response):
    email = _normalize_email(req.email)
    remote_ip = request.client.host if request.client else "unknown"
    attempt_key = auth.login_attempt_key(email, remote_ip)
    if auth.login_is_limited(attempt_key):
        raise HTTPException(429, "Too many sign-in attempts. Try again in 15 minutes.")
    result = auth.authenticate(email, req.password)
    if not result:
        auth.record_login_failure(attempt_key)
        raise HTTPException(401, "Invalid email or password.")
    user, token = result
    auth.clear_login_failures(attempt_key)
    _set_session_cookie(response, token)
    return {"user": user}


@app.get("/api/auth/me")
def current_account(user: dict = Depends(_session_user)):
    return {"user": user}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, user: dict = Depends(_session_user)):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        auth.revoke_session(token)
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, secure=bool(SESSION_COOKIE_SECURE), samesite="lax")
    return {"status": "ok"}


@app.post("/api/auth/change-password")
def change_password(req: ChangePasswordRequest, request: Request, user: dict = Depends(_session_user)):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(401, "Please sign in to continue.")
    remote_ip = request.client.host if request.client else "unknown"
    attempt_key = auth.login_attempt_key(f"change-password:{user['id']}", remote_ip)
    if auth.login_is_limited(attempt_key):
        raise HTTPException(429, "Too many password-change attempts. Try again in 15 minutes.")

    result = auth.change_password(user["id"], token, req.current_password, req.new_password)
    if result == "invalid_current_password":
        auth.record_login_failure(attempt_key)
        raise HTTPException(400, "Current password is incorrect.")
    auth.clear_login_failures(attempt_key)
    if result == "same_password":
        raise HTTPException(400, "Choose a new password that is different from your current password.")
    return {"status": "ok", "message": "Password changed. Other active sessions were signed out."}


@app.post("/api/upload")
def upload(file: UploadFile = File(...), user: dict = Depends(_session_user)):
    content = file.file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    df = read_upload(file.filename or "", content)
    ds_id = store.new_id()
    auth.record_dataset_owner(ds_id, user["id"])
    meta = {"id": ds_id, "filename": file.filename, "rows": len(df), "cols": df.shape[1],
            "uploaded_at": datetime.now().isoformat(timespec="seconds"), "target": None, "size_kb": round(len(content) / 1024, 1)}
    store.save_df(ds_id, "raw", df)
    _save_meta(ds_id, meta)
    columns = _profile(df)
    return to_jsonable({"meta": meta, "status": _status(ds_id), "columns": columns, "model_columns": columns,
                        "preview": df_records(df, 12),
                        "missing_cells": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum())})


@app.get("/api/datasets/{ds_id}")
def dataset(ds_id: str, user: dict = Depends(_owned_dataset_user)):
    meta = _meta(ds_id)
    df = store.load_df(ds_id, "raw")
    status = _status(ds_id)
    columns = _profile(df)
    model_columns = _profile(store.load_df(ds_id, "clean")) if status["cleaned"] else columns
    return to_jsonable({"meta": meta, "status": status, "columns": columns, "model_columns": model_columns,
                        "preview": df_records(df, 12),
                        "missing_cells": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum())})


@app.post("/api/datasets/{ds_id}/preprocess")
def preprocess(ds_id: str, req: PreprocessRequest, user: dict = Depends(_owned_dataset_user)):
    raw = store.load_df(ds_id, "raw")
    clean, report = run_preprocessing(raw, req.target, req.imputation, req.outlier_treatment, req.drop_duplicates,
                                      req.missing_col_threshold, req.contamination, req.categorical_encoding)
    store.save_df(ds_id, "clean", clean)
    store.save_json(ds_id, "preprocess", report)
    meta = _meta(ds_id)
    meta["target"] = req.target or meta.get("target")
    _save_meta(ds_id, meta)
    _invalidate(ds_id, "model.joblib", "ml.json", "insights.json", "predictions.csv", "report.pdf", "eda.json")
    return to_jsonable(report)


@app.get("/api/datasets/{ds_id}/preprocess")
def preprocess_result(ds_id: str, user: dict = Depends(_owned_dataset_user)):
    return store.load_json(ds_id, "preprocess")


@app.get("/api/datasets/{ds_id}/download/cleaned")
def download_cleaned(ds_id: str, format: Literal["csv", "xlsx"] = "csv", user: dict = Depends(_owned_dataset_user)):
    df = store.load_df(ds_id, "clean")
    name = safe_name(Path(_meta(ds_id)["filename"] or "dataset").stem) + "_cleaned"
    if format == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        return Response(buf.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="{name}.xlsx"'})
    return Response(df.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})


@app.get("/api/datasets/{ds_id}/eda")
def eda(ds_id: str, source: Literal["auto", "raw", "cleaned"] = "auto", target: Optional[str] = Query(None),
        user: dict = Depends(_owned_dataset_user)):
    df, used = _working_df(ds_id, source)
    meta = _meta(ds_id)
    tgt = target or meta.get("target")
    if tgt and tgt not in df.columns:
        tgt = None
    result = run_eda(df, tgt, used)
    store.save_json(ds_id, "eda", result)
    return to_jsonable(result)


@app.post("/api/datasets/{ds_id}/train")
def train(ds_id: str, req: TrainRequest, user: dict = Depends(_owned_dataset_user)):
    df, used = _working_df(ds_id)
    preprocessing = store.load_json(ds_id, "preprocess", required=False) or {}
    options = preprocessing.get("options", {})
    categorical_encoding = options.get("categorical_encoding", "onehot")
    if not isinstance(categorical_encoding, str):
        categorical_encoding = "onehot"
    bundle, results, pred_df = train_model(
        df, req.target, req.test_size, req.tune, req.n_estimators,
        categorical_encoding, req.model,
    )
    results["data_source"] = used
    store.save_model(ds_id, bundle)
    store.save_json(ds_id, "ml", results)
    pred_df.to_csv(store.folder(ds_id) / "predictions.csv", index_label="row")
    meta = _meta(ds_id)
    meta["target"] = req.target
    _save_meta(ds_id, meta)
    _invalidate(ds_id, "insights.json", "report.pdf")
    return to_jsonable(results)


@app.get("/api/datasets/{ds_id}/train")
def train_result(ds_id: str, user: dict = Depends(_owned_dataset_user)):
    return store.load_json(ds_id, "ml")


@app.post("/api/datasets/{ds_id}/predict")
def predict(ds_id: str, req: PredictRequest, user: dict = Depends(_owned_dataset_user)):
    return to_jsonable(predict_one(store.load_model(ds_id), req.values))


@app.get("/api/datasets/{ds_id}/download/predictions")
def download_predictions(ds_id: str, user: dict = Depends(_owned_dataset_user)):
    p = store.folder(ds_id) / "predictions.csv"
    if not p.exists():
        raise HTTPException(404, "Train a model first.")
    return Response(p.read_bytes(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="test_predictions.csv"'})


@app.get("/api/datasets/{ds_id}/insights")
def insights(ds_id: str, user: dict = Depends(_owned_dataset_user)):
    return to_jsonable(_insights_payload(ds_id))


@app.post("/api/datasets/{ds_id}/ask")
def ask(ds_id: str, req: AskRequest, user: dict = Depends(_owned_dataset_user)):
    ctx = {"meta": _meta(ds_id), "prep": store.load_json(ds_id, "preprocess", required=False),
           "eda": store.load_json(ds_id, "eda", required=False), "ml": store.load_json(ds_id, "ml", required=False),
           "insights": store.load_json(ds_id, "insights", required=False)}
    if ctx["insights"] is None:
        ctx["insights"] = _insights_payload(ds_id)
        ctx["eda"] = store.load_json(ds_id, "eda", required=False)
    return to_jsonable(ins_svc.answer_question(req.question, ctx))


@app.post("/api/datasets/{ds_id}/explain-row")
def explain_row(ds_id: str, req: ExplainRequest, user: dict = Depends(_owned_dataset_user)):
    return to_jsonable(ins_svc.explain_row(store.load_model(ds_id), req.row))


@app.post("/api/datasets/{ds_id}/report/generate")
def generate_report(ds_id: str, user: dict = Depends(_owned_dataset_user)):
    payload = _insights_payload(ds_id)
    meta = _meta(ds_id)
    prep = store.load_json(ds_id, "preprocess", required=False)
    ml = store.load_json(ds_id, "ml", required=False)
    eda_res = store.load_json(ds_id, "eda")
    pdf = build_pdf(meta, prep, eda_res, ml, payload, payload.get("pdp"))
    (store.folder(ds_id) / "report.pdf").write_bytes(pdf)
    sections = ["Executive summary", "Dataset overview", "Data preprocessing", "Exploratory data analysis",
                "Machine learning" if ml else "Machine learning (not trained)", "AI insights and explainability"]
    return {"filename": safe_name(Path(meta["filename"] or "dataset").stem) + "_report.pdf", "size_kb": round(len(pdf) / 1024, 1),
            "generated_at": datetime.now().isoformat(timespec="seconds"), "sections": sections,
            "includes_model": bool(ml), "includes_cleaning": bool(prep)}


@app.get("/api/datasets/{ds_id}/report/download")
def download_report(ds_id: str, user: dict = Depends(_owned_dataset_user)):
    p = store.folder(ds_id) / "report.pdf"
    if not p.exists():
        raise HTTPException(404, "Generate the report first.")
    name = safe_name(Path(_meta(ds_id)["filename"] or "dataset").stem) + "_report.pdf"
    return Response(p.read_bytes(), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{name}"'})


# Optional: serve the production build of the React app (npm run build) from the same server.
_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _dist.exists():
    class SPAStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            try:
                response = await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code != 404 or "text/html" not in Headers(scope=scope).get("accept", ""):
                    raise
                return await super().get_response("index.html", scope)
            if response.status_code == 404 and "text/html" in Headers(scope=scope).get("accept", ""):
                return await super().get_response("index.html", scope)
            return response

    app.mount("/", SPAStaticFiles(directory=_dist, html=True), name="frontend")
