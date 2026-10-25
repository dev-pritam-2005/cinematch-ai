"""Content-based recommender: TF-IDF vectors + cosine similarity.

Pipeline
--------
1. ``features`` (weighted text built in preprocessing) -> ``TfidfVectorizer`` -> sparse matrix.
   Rows are L2-normalised, so the dot product of two rows *is* their cosine similarity.
2. For one selected movie we compute cosine similarity against all movies with a single
   sparse row-vs-matrix product. The full N x N similarity matrix is never stored.
3. Candidates are filtered (genre / year), ranked by score with a deterministic
   tie-break (movie_id), and each result gets an explanation built only from fields
   that really are shared in the dataset.
"""
from __future__ import annotations

import difflib
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import normalize_words

DEFAULT_TFIDF_PARAMS: dict = {
    "max_features": 20000,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": False,
    "norm": "l2",
}
REQUIRED_COLUMNS = ("movie_id", "title", "overview", "genres", "keywords", "cast",
                    "director", "release_year", "label", "features")


class RecommenderError(Exception):
    """Base class for recommender problems."""


class MovieNotFoundError(RecommenderError):
    def __init__(self, query: object, suggestions: list[str] | None = None):
        self.query = query
        self.suggestions = suggestions or []
        hint = f" Did you mean: {', '.join(self.suggestions)}?" if self.suggestions else ""
        super().__init__(f"No movie found for {query!r}.{hint}")


class AmbiguousTitleError(RecommenderError):
    """Several different movies share this title; ``candidates`` lets the caller choose."""

    def __init__(self, title: str, candidates: pd.DataFrame):
        self.title = title
        self.candidates = candidates
        labels = ", ".join(candidates["label"].tolist())
        super().__init__(f"{title!r} matches several movies: {labels}. Select one by its id or label.")


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    return frozenset(ENGLISH_STOP_WORDS)


class MovieRecommender:
    """Fits TF-IDF on a prepared movie table and serves recommendations."""

    def __init__(self, movies: pd.DataFrame, *, vectorizer_params: dict | None = None):
        if movies is None or len(movies) == 0:
            raise RecommenderError("The movie table is empty; nothing to recommend from.")
        absent = [c for c in REQUIRED_COLUMNS if c not in movies.columns]
        if absent:
            raise RecommenderError(f"Movie table is missing column(s): {', '.join(absent)}")

        self.movies = movies.reset_index(drop=True)
        self.params = {**DEFAULT_TFIDF_PARAMS, **(vectorizer_params or {})}
        self.vectorizer, self.matrix = self._fit(self.movies["features"].fillna("").astype(str))

        self._ids = self.movies["movie_id"].to_numpy()
        self._id_to_idx = {int(m): i for i, m in enumerate(self._ids)}
        self._title_key = self.movies["title"].astype(str).str.casefold().str.strip()
        self._label_key = self.movies["label"].astype(str).str.casefold().str.strip()
        self._genre_sets = [frozenset(g.casefold() for g in gs) for gs in self.movies["genres"]]
        self._keyword_lists = [list(k) for k in self.movies["keywords"]]
        self._years = pd.to_numeric(self.movies["release_year"], errors="coerce").to_numpy(dtype=float)
        self._idf = dict(zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_))

    # ------------------------------------------------------------------ fitting
    def _fit(self, texts: pd.Series):
        params = dict(self.params)
        for attempt in range(2):
            vectorizer = TfidfVectorizer(**params)
            try:
                matrix = vectorizer.fit_transform(texts).tocsr()
                self.params = params
                return vectorizer, matrix
            except ValueError as exc:  # "empty vocabulary" on tiny / featureless data
                if attempt == 0 and params.get("min_df", 1) != 1:
                    params["min_df"] = 1
                    continue
                raise RecommenderError(f"Could not build TF-IDF features: {exc}") from exc
        raise RecommenderError("Could not build TF-IDF features.")  # pragma: no cover

    # ------------------------------------------------------------------ lookups
    def __len__(self) -> int:
        return len(self.movies)

    @property
    def all_genres(self) -> list[str]:
        names: dict[str, str] = {}
        for gs in self.movies["genres"]:
            for g in gs:
                names.setdefault(g.casefold(), g)
        return sorted(names.values(), key=str.casefold)

    @property
    def year_bounds(self) -> tuple[int, int] | None:
        valid = self._years[~np.isnan(self._years)]
        return (int(valid.min()), int(valid.max())) if valid.size else None

    def index_for_id(self, movie_id: int) -> int:
        try:
            return self._id_to_idx[int(movie_id)]
        except (KeyError, TypeError, ValueError):
            raise MovieNotFoundError(movie_id) from None

    def search(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Case-insensitive title search; prefix matches first, then substring, then votes."""
        q = (query or "").casefold().strip()
        if not q or limit <= 0:
            return self.movies.iloc[0:0].copy()
        starts = self._title_key.str.startswith(q, na=False)
        contains = self._title_key.str.contains(q, regex=False, na=False)
        hit = contains | self._label_key.str.contains(q, regex=False, na=False)
        if not hit.any():
            return self.movies.iloc[0:0].copy()
        found = self.movies[hit].assign(_rank=(~starts[hit]).astype(int))
        found = found.sort_values(["_rank", "vote_count", "movie_id"], ascending=[True, False, True], kind="stable")
        return found.drop(columns="_rank").head(limit).reset_index(drop=True)

    def resolve_title(self, title: str) -> int:
        """Row index for an exact title/label. Raises NotFound / Ambiguous otherwise."""
        key = (title or "").casefold().strip() if isinstance(title, str) else ""
        if not key:
            raise MovieNotFoundError(title)
        label_hits = np.flatnonzero((self._label_key == key).to_numpy())
        if label_hits.size == 1:
            return int(label_hits[0])
        title_hits = np.flatnonzero((self._title_key == key).to_numpy())
        if title_hits.size == 1:
            return int(title_hits[0])
        if title_hits.size > 1:
            raise AmbiguousTitleError(title, self.movies.iloc[title_hits][["movie_id", "label"]].reset_index(drop=True))
        titles = self.movies["title"].astype(str).tolist()
        close = difflib.get_close_matches(title.strip(), titles, n=3, cutoff=0.6)
        if not close:
            close = [t for t in titles if key in t.casefold()][:3]
        raise MovieNotFoundError(title, list(dict.fromkeys(close)))

    # ------------------------------------------------------------------ scoring
    def similarity_scores(self, idx: int) -> np.ndarray:
        """Cosine similarity of one movie to every movie (dense vector, length N)."""
        return cosine_similarity(self.matrix[idx], self.matrix, dense_output=True).ravel()

    def recommend(
        self,
        movie_title: str | None = None,
        top_n: int = 10,
        *,
        movie_id: int | None = None,
        genres: Sequence[str] | None = None,
        year_range: tuple[int, int] | None = None,
        explain: bool = True,
    ) -> pd.DataFrame:
        """Top-N similar movies. Raises MovieNotFoundError / AmbiguousTitleError / ValueError.

        ``genres`` keeps movies having *any* of the given genres; ``year_range`` is inclusive.
        Movies with an unknown release year are excluded only when ``year_range`` is given.
        """
        if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        if movie_id is not None:
            idx = self.index_for_id(movie_id)
        elif movie_title is not None:
            idx = self.resolve_title(movie_title)
        else:
            raise ValueError("Provide movie_title or movie_id.")

        scores = np.round(self.similarity_scores(idx), 10)  # rounding makes ties exact & deterministic
        mask = scores > 0
        mask[idx] = False  # never recommend the movie to itself

        if genres:
            wanted = {g.casefold() for g in genres}
            mask &= np.fromiter((bool(gs & wanted) for gs in self._genre_sets), dtype=bool, count=len(self))
        if year_range is not None:
            lo, hi = sorted(int(y) for y in year_range)
            with np.errstate(invalid="ignore"):
                mask &= (self._years >= lo) & (self._years <= hi)

        cand = np.flatnonzero(mask)
        order = np.lexsort((self._ids[cand], -scores[cand]))  # score desc, then movie_id asc
        chosen = cand[order]

        # distinctness: one entry per movie id (guaranteed) and per (title, year) pair
        seen: set[tuple[str, object]] = set()
        picked: list[int] = []
        for i in chosen:
            key = (self._title_key.iat[i], self._years[i])
            if key in seen:
                continue
            seen.add(key)
            picked.append(int(i))
            if len(picked) >= top_n:
                break

        result = self.movies.iloc[picked].copy()
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        result["similarity_score"] = scores[picked].astype(float)
        if explain:
            result["explanation"] = [self.explain(idx, i) for i in picked]
        result.attrs["seed_movie_id"] = int(self._ids[idx])
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------ explanations
    def _overview_terms(self, idx: int) -> set[str]:
        stop = _stop_words()
        return {w for w in normalize_words(self.movies["overview"].iat[idx]) if len(w) > 3 and w not in stop}

    def explain(self, seed_idx: int, cand_idx: int) -> str:
        """Short explanation containing only attributes verifiably shared by both movies."""
        seed, cand = self.movies.iloc[seed_idx], self.movies.iloc[cand_idx]
        parts: list[str] = []

        shared_genres = [g for g in cand["genres"] if g.casefold() in self._genre_sets[seed_idx]]
        if shared_genres:
            parts.append("Shared genres: " + ", ".join(shared_genres))

        seed_kw = {k.casefold() for k in self._keyword_lists[seed_idx]}
        shared_kw = [k for k in self._keyword_lists[cand_idx] if k.casefold() in seed_kw]
        if shared_kw:
            more = f" (+{len(shared_kw) - 4} more)" if len(shared_kw) > 4 else ""
            parts.append("Shared keywords: " + ", ".join(shared_kw[:4]) + more)

        if seed["director"] and seed["director"].casefold() == str(cand["director"]).casefold():
            parts.append(f"Same director: {seed['director']}")

        seed_cast = {c.casefold() for c in seed["cast"]}
        shared_cast = [c for c in cand["cast"] if c.casefold() in seed_cast]
        if shared_cast:
            parts.append("Shared cast: " + ", ".join(shared_cast))

        overlap = (self._overview_terms(seed_idx) & self._overview_terms(cand_idx)) & self._idf.keys()
        if overlap:
            top_terms = sorted(overlap, key=lambda w: (-self._idf[w], w))[:4]
            parts.append("Similar descriptions mention: " + ", ".join(top_terms))

        if not parts:
            return "Only a weak overlap in combined text features; no shared genre, keyword, cast or director."
        return " · ".join(parts)


# ----------------------------------------------------------------------- convenience API
@lru_cache(maxsize=1)
def get_default_recommender() -> MovieRecommender:
    """Build a recommender from the configured data directory (cached per process)."""
    from .data_loader import load_raw_data
    from .preprocessing import build_movie_table

    movies_raw, credits_raw = load_raw_data()
    movies, _ = build_movie_table(movies_raw, credits_raw)
    return MovieRecommender(movies)


def recommend_movies(
    movie_title: str,
    top_n: int = 10,
    *,
    recommender: MovieRecommender | None = None,
    genres: Iterable[str] | None = None,
    year_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Reusable entry point: ``recommend_movies("Avatar", top_n=10)``.

    Returns a DataFrame with metadata, ``similarity_score`` (cosine similarity of TF-IDF
    vectors, NOT a probability of enjoyment) and an ``explanation`` column.
    """
    rec = recommender or get_default_recommender()
    return rec.recommend(movie_title, top_n, genres=list(genres) if genres else None, year_range=year_range)
