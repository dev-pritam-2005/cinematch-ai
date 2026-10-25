"""Locate, read and validate the TMDB 5000 CSV files.

Nothing in this module invents data: if the files are missing or malformed a
descriptive exception is raised so the UI can show a setup message.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

MOVIES_FILENAME = "tmdb_5000_movies.csv"
CREDITS_FILENAME = "tmdb_5000_credits.csv"
DATA_DIR_ENV = "CINEMATCH_DATA_DIR"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

REQUIRED_MOVIE_COLUMNS = (
    "id", "title", "overview", "genres", "keywords",
    "release_date", "vote_average", "vote_count",
)
REQUIRED_CREDIT_COLUMNS = ("movie_id", "cast", "crew")


class DatasetError(Exception):
    """Base class for dataset problems that should be shown to the user."""


class DatasetNotFoundError(DatasetError):
    """One or both CSV files could not be found."""

    def __init__(self, data_dir: Path, missing: list[str]):
        self.data_dir = Path(data_dir)
        self.missing = list(missing)
        super().__init__(
            f"Missing dataset file(s) {', '.join(self.missing)} in '{self.data_dir}'."
        )


class DatasetFormatError(DatasetError):
    """A CSV file exists but cannot be parsed or lacks required columns."""


def resolve_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    """Explicit argument > CINEMATCH_DATA_DIR environment variable > ./data."""
    if data_dir:
        return Path(data_dir).expanduser()
    env_value = os.environ.get(DATA_DIR_ENV, "").strip()
    return Path(env_value).expanduser() if env_value else DEFAULT_DATA_DIR


def missing_files(data_dir: str | os.PathLike | None = None) -> list[str]:
    base = resolve_data_dir(data_dir)
    return [n for n in (MOVIES_FILENAME, CREDITS_FILENAME) if not (base / n).is_file()]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    except pd.errors.EmptyDataError as exc:
        raise DatasetFormatError(f"'{path.name}' is empty.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise DatasetFormatError(f"'{path.name}' is not a valid CSV file: {exc}") from exc


def validate_raw_frames(movies: pd.DataFrame, credits: pd.DataFrame) -> None:
    """Raise DatasetFormatError when required columns are missing or frames are empty."""
    problems = []
    for name, frame, required in (
        (MOVIES_FILENAME, movies, REQUIRED_MOVIE_COLUMNS),
        (CREDITS_FILENAME, credits, REQUIRED_CREDIT_COLUMNS),
    ):
        absent = [c for c in required if c not in frame.columns]
        if absent:
            problems.append(f"{name} is missing column(s): {', '.join(absent)}")
        elif frame.empty:
            problems.append(f"{name} contains no rows")
    if problems:
        raise DatasetFormatError("; ".join(problems))


def load_raw_data(data_dir: str | os.PathLike | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both CSV files and validate their structure."""
    base = resolve_data_dir(data_dir)
    absent = missing_files(base)
    if absent:
        raise DatasetNotFoundError(base, absent)
    movies = _read_csv(base / MOVIES_FILENAME)
    credits = _read_csv(base / CREDITS_FILENAME)
    validate_raw_frames(movies, credits)
    return movies, credits
