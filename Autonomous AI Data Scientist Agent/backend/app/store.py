"""Tiny file based store: every uploaded dataset gets its own folder under backend/storage/."""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException

from .utils import to_jsonable

STORAGE = Path(os.environ.get("MLAPP_STORAGE", Path(__file__).resolve().parent.parent / "storage"))
STORAGE.mkdir(parents=True, exist_ok=True)
_ID_RE = re.compile(r"^[a-f0-9]{12}$")


def new_id() -> str:
    ds_id = uuid.uuid4().hex[:12]
    (STORAGE / ds_id).mkdir(parents=True, exist_ok=True)
    return ds_id


def folder(ds_id: str) -> Path:
    if not _ID_RE.match(ds_id or ""):
        raise HTTPException(400, "Invalid dataset id.")
    p = STORAGE / ds_id
    if not p.exists():
        raise HTTPException(404, "Dataset not found. Please upload it again on the Dataset Upload page.")
    return p


def save_df(ds_id: str, name: str, df: pd.DataFrame) -> None:
    df.to_pickle(folder(ds_id) / f"{name}.pkl")


def has(ds_id: str, name: str) -> bool:
    return (folder(ds_id) / name).exists()


def load_df(ds_id: str, name: str) -> pd.DataFrame:
    p = folder(ds_id) / f"{name}.pkl"
    if not p.exists():
        raise HTTPException(404, f"'{name}' data is not available yet. Complete the previous step first.")
    return pd.read_pickle(p)


def save_json(ds_id: str, key: str, obj: Any) -> None:
    with open(folder(ds_id) / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f)


def load_json(ds_id: str, key: str, required: bool = True) -> Any:
    p = folder(ds_id) / f"{key}.json"
    if not p.exists():
        if required:
            raise HTTPException(404, f"'{key}' results are not available yet. Complete the previous step first.")
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model(ds_id: str, bundle: Any) -> None:
    joblib.dump(bundle, folder(ds_id) / "model.joblib")


def load_model(ds_id: str) -> Any:
    p = folder(ds_id) / "model.joblib"
    if not p.exists():
        raise HTTPException(404, "No trained model yet. Train the Random Forest on the Machine Learning page first.")
    return joblib.load(p)
