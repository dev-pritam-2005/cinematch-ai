"""Turn the raw TMDB CSVs into one clean movie table plus a model-ready text feature.

Design notes
------------
* Raw JSON-like columns are parsed defensively (``safe_json_loads``); malformed
  cells become empty lists and are *counted*, never silently invented.
* Display metadata (genres, cast, ...) is kept in its own columns. The
  ``features`` column is a separate, derived string used only by the model.
* Feature weighting is explicit and reproducible: ``FeatureWeights`` says how
  many times each field's tokens are repeated in the combined text (see README,
  "Feature weighting").
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_MALFORMED = "malformed"

LANGUAGE_NAMES = {  # display names for ISO codes; unknown codes are shown upper-cased
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "cn": "Cantonese", "hi": "Hindi",
    "ru": "Russian", "pt": "Portuguese", "sv": "Swedish", "da": "Danish", "nl": "Dutch",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish", "ar": "Arabic",
    "pl": "Polish", "he": "Hebrew", "fa": "Persian", "cs": "Czech", "no": "Norwegian",
    "fi": "Finnish", "el": "Greek", "hu": "Hungarian", "id": "Indonesian", "ro": "Romanian",
}

FINAL_COLUMNS = [
    "movie_id", "title", "original_title", "overview", "genres", "keywords", "cast",
    "director", "release_date", "release_year", "vote_average", "vote_count", "runtime",
    "original_language", "popularity", "poster_path", "label", "features",
]


# --------------------------------------------------------------------------- parsing
def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def safe_json_loads(value: object) -> tuple[list, str]:
    """Parse a JSON-ish cell into a list. Returns ``(items, status)``.

    status is ``ok`` / ``missing`` (null or blank cell) / ``malformed``.
    Python-literal style strings (single quotes) are accepted as a fallback.
    """
    if isinstance(value, list):
        return value, STATUS_OK
    if _is_missing(value):
        return [], STATUS_MISSING
    if not isinstance(value, str):
        return [], STATUS_MALFORMED
    text = value.strip()
    if not text:
        return [], STATUS_MISSING
    try:
        parsed = json.loads(text)
    except ValueError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return [], STATUS_MALFORMED
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [], STATUS_MALFORMED
    return parsed, STATUS_OK


def parse_json_list(value: object) -> list:
    return safe_json_loads(value)[0]


def extract_names(items: list, limit: int | None = None) -> list[str]:
    """Return unique ``name`` values (order preserved), skipping unusable entries."""
    if limit is not None and limit <= 0:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str):
            continue
        name = " ".join(name.split())
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        names.append(name)
        if limit is not None and len(names) >= limit:
            break
    return names


def extract_cast(items: list, limit: int = 3) -> list[str]:
    """Top-billed cast members (by TMDB ``order``), at most ``limit``."""
    def order(item: dict) -> float:
        try:
            value = float(item.get("order"))
            return value if value == value else float("inf")
        except (TypeError, ValueError):
            return float("inf")

    dicts = [i for i in items if isinstance(i, dict)]
    dicts.sort(key=order)  # stable: keeps file order for ties/missing
    return extract_names(dicts, limit=max(int(limit), 0))


def extract_director(crew: list) -> str:
    """First crew member whose job is 'Director'; empty string if none."""
    for item in crew:
        if isinstance(item, dict) and str(item.get("job", "")).strip().casefold() == "director":
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                return " ".join(name.split())
    return ""


# ------------------------------------------------------------------ text normalisation
def normalize_words(text: object) -> list[str]:
    """Lower-case, strip punctuation, split into words."""
    if not isinstance(text, str):
        return []
    return re.sub(r"[\W_]+", " ", text.casefold()).split()


def to_token(text: str) -> str:
    """'Science Fiction' -> 'science_fiction' so multi-word names stay one feature."""
    return "_".join(normalize_words(text))


@dataclass(frozen=True)
class FeatureWeights:
    """How many times each field's tokens are repeated in the combined text.

    TF-IDF term frequency then scales the weight of each field. Defaults put the most
    weight on genres and keywords (strong, curated signals) and lighter weight on the
    free-text overview, cast and director.
    """
    genres: int = 3
    keywords: int = 2
    director: int = 2
    cast: int = 1
    overview: int = 1

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def build_feature_text(
    *, genres: list[str], keywords: list[str], director: str, cast: list[str],
    overview: str, weights: FeatureWeights = FeatureWeights(),
) -> str:
    """Combine all content fields into one weighted text string for TF-IDF."""
    genre_tokens = [t for t in (to_token(g) for g in genres) if t]
    keyword_words = [w for k in keywords for w in normalize_words(k)]
    director_tokens = [t for t in [to_token(director)] if t]
    cast_tokens = [t for t in (to_token(c) for c in cast) if t]
    overview_words = normalize_words(overview)
    parts = (
        genre_tokens * max(weights.genres, 0)
        + keyword_words * max(weights.keywords, 0)
        + director_tokens * max(weights.director, 0)
        + cast_tokens * max(weights.cast, 0)
        + overview_words * max(weights.overview, 0)
    )
    return " ".join(parts)


# ------------------------------------------------------------------------- reporting
@dataclass
class DataQualityReport:
    movies_raw_rows: int = 0
    credits_raw_rows: int = 0
    rows_final: int = 0
    rows_dropped_bad_id: int = 0
    duplicate_ids_removed: int = 0
    duplicate_title_date_removed: int = 0
    rows_dropped_no_title: int = 0
    rows_without_credits: int = 0
    invalid_release_dates: int = 0
    missing_release_date: int = 0
    missing_overview: int = 0
    empty_genres: int = 0
    empty_keywords: int = 0
    missing_director: int = 0
    missing_runtime: int = 0
    unrated_movies: int = 0
    titles_shared_by_different_movies: int = 0
    malformed_json: dict[str, int] = field(default_factory=dict)
    raw_missing_movies: dict[str, int] = field(default_factory=dict)
    raw_missing_credits: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------- main pipeline
def _parse_column(series: pd.Series, name: str, report: DataQualityReport) -> list[list]:
    parsed, malformed = [], 0
    for value in series:
        items, status = safe_json_loads(value)
        malformed += status == STATUS_MALFORMED
        parsed.append(items)
    report.malformed_json[name] = int(malformed)
    return parsed


def build_movie_table(
    movies_raw: pd.DataFrame,
    credits_raw: pd.DataFrame,
    *,
    cast_size: int = 3,
    weights: FeatureWeights = FeatureWeights(),
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Merge, clean and enrich the raw frames. Returns ``(movies, quality_report)``."""
    report = DataQualityReport(
        movies_raw_rows=len(movies_raw),
        credits_raw_rows=len(credits_raw),
        raw_missing_movies={c: int(v) for c, v in movies_raw.isna().sum().items()},
        raw_missing_credits={c: int(v) for c, v in credits_raw.isna().sum().items()},
    )

    movies = movies_raw.copy()
    movies["_id"] = pd.to_numeric(movies["id"], errors="coerce")
    bad = movies["_id"].isna()
    report.rows_dropped_bad_id = int(bad.sum())
    movies = movies[~bad].copy()
    movies["movie_id"] = movies["_id"].astype("int64")
    dup_ids = movies.duplicated("movie_id", keep="first")
    report.duplicate_ids_removed = int(dup_ids.sum())
    movies = movies[~dup_ids]

    credits = credits_raw[["movie_id", "cast", "crew"]].copy()
    credits["movie_id"] = pd.to_numeric(credits["movie_id"], errors="coerce")
    credits = credits.dropna(subset=["movie_id"])
    credits["movie_id"] = credits["movie_id"].astype("int64")
    credits = credits.drop_duplicates("movie_id", keep="first")

    # Left join: a movie without a credits row is kept (empty cast/director) and counted.
    report.rows_without_credits = int((~movies["movie_id"].isin(credits["movie_id"])).sum())
    df = movies.merge(credits, on="movie_id", how="left", suffixes=("", "_credits"))
    df["_order"] = np.arange(len(df))

    # ---- titles
    if "original_title" not in df.columns:
        df["original_title"] = ""
    title = df["title"].where(df["title"].notna(), df["original_title"]).fillna("").astype(str)
    df["title"] = title.str.strip()
    no_title = df["title"] == ""
    report.rows_dropped_no_title = int(no_title.sum())
    df = df[~no_title].copy()
    df["original_title"] = df["original_title"].fillna("").astype(str).str.strip()

    # ---- text
    df["overview"] = df["overview"].fillna("").astype(str).str.strip()
    report.missing_overview = int((df["overview"] == "").sum())

    # ---- JSON-like columns
    df["genres"] = pd.Series([extract_names(x) for x in _parse_column(df["genres"], "genres", report)],
                             index=df.index, dtype=object)
    df["keywords"] = pd.Series([extract_names(x) for x in _parse_column(df["keywords"], "keywords", report)],
                               index=df.index, dtype=object)
    cast_items = _parse_column(df["cast"], "cast", report)
    crew_items = _parse_column(df["crew"], "crew", report)
    df["cast"] = pd.Series([extract_cast(x, cast_size) for x in cast_items], index=df.index, dtype=object)
    df["director"] = [extract_director(x) for x in crew_items]
    report.empty_genres = int(df["genres"].map(len).eq(0).sum())
    report.empty_keywords = int(df["keywords"].map(len).eq(0).sum())
    report.missing_director = int((df["director"] == "").sum())

    # ---- dates
    raw_dates = df["release_date"]
    blank = raw_dates.isna() | (raw_dates.astype(str).str.strip() == "")
    df["release_date"] = pd.to_datetime(raw_dates, errors="coerce")
    report.missing_release_date = int(blank.sum())
    report.invalid_release_dates = int((df["release_date"].isna() & ~blank).sum())
    df["release_year"] = df["release_date"].dt.year.astype("Int64")

    # ---- numbers (0 votes means "unrated", not "rated 0")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).clip(lower=0).astype("int64")
    rating = pd.to_numeric(df["vote_average"], errors="coerce")
    rating = rating.where((rating >= 0) & (rating <= 10))
    df["vote_average"] = rating.where(df["vote_count"] > 0)
    report.unrated_movies = int(df["vote_average"].isna().sum())
    nan_series = pd.Series(np.nan, index=df.index)
    runtime = pd.to_numeric(df["runtime"], errors="coerce") if "runtime" in df.columns else nan_series
    df["runtime"] = runtime.where(runtime > 0)
    report.missing_runtime = int(df["runtime"].isna().sum())
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce") if "popularity" in df.columns else nan_series
    if "original_language" in df.columns:
        df["original_language"] = df["original_language"].fillna("").astype(str).str.strip()
    else:
        df["original_language"] = ""

    # ---- poster path: the TMDB 5000 CSVs have none, but honour the column if a user adds it
    if "poster_path" in df.columns:
        df["poster_path"] = pd.Series(
            [p.strip() if isinstance(p, str) and p.strip() else None for p in df["poster_path"]],
            index=df.index, dtype=object)
    else:
        df["poster_path"] = pd.Series([None] * len(df), index=df.index, dtype=object)

    # ---- same title AND same release date => duplicate record; keep the one with most votes
    df["_tkey"] = df["title"].str.casefold()
    df = df.sort_values(["vote_count", "movie_id"], ascending=[False, True], kind="stable")
    dup_td = df.duplicated(["_tkey", "release_date"], keep="first") & df["release_date"].notna()
    report.duplicate_title_date_removed = int(dup_td.sum())
    df = df[~dup_td].sort_values("_order", kind="stable")

    # ---- unique, human-readable labels ("Batman (1989)")
    year_txt = df["release_year"].map(lambda y: f" ({int(y)})" if pd.notna(y) else "")
    label = df["title"] + year_txt
    clash = label.duplicated(keep=False)
    df["label"] = label.where(~clash, label + " · #" + df["movie_id"].astype(str))
    report.titles_shared_by_different_movies = int(df["_tkey"].duplicated(keep=False).sum())

    # ---- model feature text
    df["features"] = [
        build_feature_text(genres=g, keywords=k, director=d, cast=c, overview=o, weights=weights)
        for g, k, d, c, o in zip(df["genres"], df["keywords"], df["director"], df["cast"], df["overview"])
    ]

    out = df[FINAL_COLUMNS].reset_index(drop=True)
    report.rows_final = len(out)
    return out, report


def language_name(code: str) -> str:
    code = (code or "").strip()
    return LANGUAGE_NAMES.get(code.casefold(), code.upper()) if code else ""
