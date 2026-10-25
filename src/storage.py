"""Local, single-user persistence (SQLite) for watchlist, favourites and personal ratings.

This is intended for a local demo. A public deployment sharing one database file would
expose one visitor's saved movies to everyone; add authentication and per-user rows
before using it that way (see README).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .data_loader import PROJECT_ROOT

DB_PATH_ENV = "CINEMATCH_DB_PATH"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "cinematch.db"
KINDS = ("watchlist", "favorite")


def resolve_db_path(path: str | os.PathLike | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    env_value = os.environ.get(DB_PATH_ENV, "").strip()
    return Path(env_value).expanduser() if env_value else DEFAULT_DB_PATH


class LocalStore:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = resolve_db_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS collection (
                    movie_id INTEGER NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('watchlist','favorite')),
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (movie_id, kind)
                );
                CREATE TABLE IF NOT EXISTS ratings (
                    movie_id INTEGER PRIMARY KEY,
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    updated_at TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        # Short-lived connections avoid cross-thread problems with Streamlit's script threads.
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _check_kind(kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    # ---- watchlist / favourites
    def add(self, movie_id: int, kind: str) -> None:
        self._check_kind(kind)
        with closing(self._connect()) as con, con:
            con.execute("INSERT OR IGNORE INTO collection VALUES (?,?,?)", (int(movie_id), kind, self._now()))

    def remove(self, movie_id: int, kind: str) -> None:
        self._check_kind(kind)
        with closing(self._connect()) as con, con:
            con.execute("DELETE FROM collection WHERE movie_id=? AND kind=?", (int(movie_id), kind))

    def contains(self, movie_id: int, kind: str) -> bool:
        self._check_kind(kind)
        with closing(self._connect()) as con:
            row = con.execute("SELECT 1 FROM collection WHERE movie_id=? AND kind=?", (int(movie_id), kind)).fetchone()
        return row is not None

    def toggle(self, movie_id: int, kind: str) -> bool:
        """Add if absent, remove if present. Returns True when the movie is now saved."""
        if self.contains(movie_id, kind):
            self.remove(movie_id, kind)
            return False
        self.add(movie_id, kind)
        return True

    def list_ids(self, kind: str) -> list[int]:
        """Saved movie ids, most recently added first."""
        self._check_kind(kind)
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT movie_id FROM collection WHERE kind=? ORDER BY added_at DESC, rowid DESC", (kind,)
            ).fetchall()
        return [r[0] for r in rows]

    # ---- personal ratings (1-5)
    def set_rating(self, movie_id: int, rating: int) -> None:
        if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
            raise ValueError("rating must be an integer from 1 to 5")
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT INTO ratings VALUES (?,?,?) ON CONFLICT(movie_id) DO UPDATE SET "
                "rating=excluded.rating, updated_at=excluded.updated_at",
                (int(movie_id), rating, self._now()),
            )

    def get_rating(self, movie_id: int) -> int | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT rating FROM ratings WHERE movie_id=?", (int(movie_id),)).fetchone()
        return row[0] if row else None

    def clear_rating(self, movie_id: int) -> None:
        with closing(self._connect()) as con, con:
            con.execute("DELETE FROM ratings WHERE movie_id=?", (int(movie_id),))

    def all_ratings(self) -> dict[int, int]:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT movie_id, rating FROM ratings ORDER BY updated_at DESC, rowid DESC").fetchall()
        return {r[0]: r[1] for r in rows}
