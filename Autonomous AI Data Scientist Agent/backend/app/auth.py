"""Local account, session, and dataset ownership storage."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(os.environ.get(
    "MLAPP_AUTH_DB",
    Path(__file__).resolve().parent.parent / "storage" / "auth.sqlite3",
))
DATASET_STORAGE = Path(os.environ.get(
    "MLAPP_STORAGE",
    Path(__file__).resolve().parent.parent / "storage",
))
SESSION_SECONDS = 14 * 24 * 60 * 60
PASSWORD_ITERATIONS = 600_000
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_ATTEMPT_LIMIT = 10


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
            CREATE TABLE IF NOT EXISTS dataset_owners (
                dataset_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS login_attempts (
                attempt_key TEXT NOT NULL,
                attempted_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS login_attempt_key_time
                ON login_attempts(attempt_key, attempted_at);
            """
        )


def _password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


DUMMY_PASSWORD_HASH = _password_hash(secrets.token_urlsafe(24))


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = encoded.split("$")
        if algorithm != "pbkdf2_sha256" or int(iterations) != PASSWORD_ITERATIONS:
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        if len(salt) != 16 or len(expected) != 32:
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def _new_session(conn: sqlite3.Connection, user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    conn.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (hashlib.sha256(token.encode("ascii")).hexdigest(), user_id, now + SESSION_SECONDS, now),
    )
    return token


def create_user(name: str, email: str, password: str, legacy_dataset_id: str | None = None) -> tuple[dict[str, Any], str, str | None]:
    user_id = uuid.uuid4().hex
    now = int(time.time())
    adopted_dataset_id = None
    with _connect() as conn:
        is_first_user = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"] == 0
        conn.execute(
            "INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, name, email, _password_hash(password), now),
        )
        if (
            is_first_user
            and legacy_dataset_id
            and re.fullmatch(r"[a-f0-9]{12}", legacy_dataset_id)
            and (DATASET_STORAGE / legacy_dataset_id / "meta.json").is_file()
        ):
            conn.execute(
                "INSERT INTO dataset_owners (dataset_id, user_id) VALUES (?, ?)",
                (legacy_dataset_id, user_id),
            )
            adopted_dataset_id = legacy_dataset_id
        token = _new_session(conn, user_id)
    return {"id": user_id, "name": name, "email": email}, token, adopted_dataset_id


def authenticate(email: str, password: str) -> tuple[dict[str, Any], str] | None:
    with _connect() as conn:
        user = conn.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        stored_hash = user["password_hash"] if user else DUMMY_PASSWORD_HASH
        password_matches = verify_password(password, stored_hash)
        if user is None or not password_matches:
            return None
        token = _new_session(conn, user["id"])
    return {"id": user["id"], "name": user["name"], "email": user["email"]}, token


def change_password(
    user_id: str, current_session_token: str, current_password: str, new_password: str
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", current_session_token):
        raise ValueError("A valid current session token is required.")
    current_token_hash = hashlib.sha256(current_session_token.encode("ascii")).hexdigest()
    with _connect() as conn:
        user = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if user is None or not verify_password(current_password, user["password_hash"]):
            return "invalid_current_password"
        if verify_password(new_password, user["password_hash"]):
            return "same_password"
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (_password_hash(new_password), user_id),
        )
        conn.execute(
            "DELETE FROM sessions WHERE user_id = ? AND token_hash != ?",
            (user_id, current_token_hash),
        )
    return "ok"


def get_session_user(token: str) -> dict[str, Any] | None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
        return None
    token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    now = int(time.time())
    with _connect() as conn:
        row = conn.execute(
            """SELECT users.id, users.name, users.email
               FROM sessions JOIN users ON users.id = sessions.user_id
               WHERE sessions.token_hash = ? AND sessions.expires_at > ?""",
            (token_hash, now),
        ).fetchone()
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
    return dict(row) if row else None


def revoke_session(token: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
        return
    token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


def record_dataset_owner(dataset_id: str, user_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO dataset_owners (dataset_id, user_id) VALUES (?, ?)",
            (dataset_id, user_id),
        )


def owns_dataset(dataset_id: str, user_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM dataset_owners WHERE dataset_id = ? AND user_id = ?",
            (dataset_id, user_id),
        ).fetchone()
    return row is not None


def login_attempt_key(email: str, remote_ip: str) -> str:
    return hashlib.sha256(f"{email}\0{remote_ip}".encode("utf-8")).hexdigest()


def login_is_limited(attempt_key: str) -> bool:
    now = int(time.time())
    with _connect() as conn:
        conn.execute("DELETE FROM login_attempts WHERE attempted_at <= ?", (now - LOGIN_WINDOW_SECONDS,))
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM login_attempts WHERE attempt_key = ? AND attempted_at > ?",
            (attempt_key, now - LOGIN_WINDOW_SECONDS),
        ).fetchone()
    return row["count"] >= LOGIN_ATTEMPT_LIMIT


def record_login_failure(attempt_key: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO login_attempts (attempt_key, attempted_at) VALUES (?, ?)",
            (attempt_key, int(time.time())),
        )


def clear_login_failures(attempt_key: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM login_attempts WHERE attempt_key = ?", (attempt_key,))


init_db()
