"""Reads uploaded CSV / Excel files into a DataFrame."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

MAX_ROWS = 200_000
ALLOWED = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"}


def read_upload(filename: str, content: bytes) -> pd.DataFrame:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type '{ext or 'unknown'}'. Upload a .csv, .tsv, .xlsx or .xls file.")
    if not content:
        raise HTTPException(400, "The uploaded file is empty.")

    df = None
    try:
        if ext in {".xlsx", ".xlsm", ".xls"}:
            df = pd.read_excel(io.BytesIO(content), sheet_name=0)
        else:
            last_err = None
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=enc)
                    break
                except UnicodeDecodeError as e:  # try the next encoding
                    last_err = e
            if df is None:
                raise ValueError(f"Could not decode file: {last_err}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read the file: {e}")

    if df.empty or df.shape[1] == 0:
        raise HTTPException(400, "The file has no data rows or columns.")
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42).sort_index().reset_index(drop=True)

    # clean headers: strings, stripped, unique, never empty
    cols, seen = [], {}
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"column_{i + 1}"
        if name.lower().startswith("unnamed:"):
            name = f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df.columns = cols
    return df.reset_index(drop=True)
