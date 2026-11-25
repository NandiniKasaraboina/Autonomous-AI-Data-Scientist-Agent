"""Small helpers shared by every service."""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(o: Any) -> Any:
    """Recursively convert numpy / pandas objects into plain JSON-safe values (NaN -> None)."""
    if o is None or o is pd.NaT or o is pd.NA:
        return None
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, pd.Series):
        return to_jsonable(o.tolist())
    return o if isinstance(o, str) else str(o)


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def is_categorical(s: pd.Series) -> bool:
    return not is_numeric(s) and not is_datetime(s)


def df_records(df: pd.DataFrame, n: int = 10) -> list[dict]:
    head = df.head(n).astype(object)
    return to_jsonable(head.where(head.notna(), None).to_dict("records"))


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:80] or "dataset"
