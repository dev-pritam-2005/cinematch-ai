"""Streamlit helpers shared by all pages: cached engine, session state, CSS and movie cards."""
from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

from . import tmdb_client
from .data_loader import (CREDITS_FILENAME, MOVIES_FILENAME, DatasetError, DatasetFormatError,
                          DatasetNotFoundError, resolve_data_dir)
from .preprocessing import DataQualityReport, build_movie_table
from .data_loader import load_raw_data
from .recommender import MovieRecommender
from .storage import LocalStore, resolve_db_path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = ROOT / "assets" / "placeholder.svg"
KAGGLE_URL = "https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata"

PAGE_HOME = "pages/0_Home.py"
PAGE_SIMILAR = "pages/1_Find_Similar_Movies.py"
PAGE_EXPLORER = "pages/2_Movie_Explorer.py"
PAGE_DETAILS = "pages/3_Movie_Details.py"
PAGE_MOOD = "pages/4_Mood_Discovery.py"
PAGE_WATCHLIST = "pages/5_Watchlist.py"
PAGE_INSIGHTS = "pages/6_ML_Insights.py"
PAGE_ABOUT = "pages/7_About.py"

SIMILARITY_HELP = ("Cosine similarity between TF-IDF vectors of the two movies (0 = nothing in common, 1 = identical "
                   "content features). It describes content overlap only - it is not a probability that you will enjoy the movie.")

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
.cm-hero {padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
  background: radial-gradient(circle at 15% 20%, rgba(139,92,246,.35), transparent 55%),
              radial-gradient(circle at 90% 80%, rgba(34,211,238,.18), transparent 50%), #14151d;
  border: 1px solid rgba(255,255,255,.08);}
.cm-hero h1 {margin: 0 0 .3rem 0; font-size: 2.3rem; letter-spacing: -.02em;}
.cm-hero p {margin: 0; color: #b6b9cc; max-width: 46rem;}
.cm-accent {background: linear-gradient(90deg,#a78bfa,#22d3ee); -webkit-background-clip: text; background-clip: text; color: transparent;}
.cm-poster {aspect-ratio: 2/3; width: 100%; border-radius: 10px; overflow: hidden; background: #14151d;}
.cm-poster img {width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .25s ease;}
.cm-poster:hover img {transform: scale(1.04);}
.cm-title {font-weight: 600; margin: .55rem 0 .1rem; line-height: 1.25; min-height: 2.5em; overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;}
.cm-meta {color: #a3a6bd; font-size: .82rem;}
.cm-chip {display: inline-block; padding: 1px 8px; margin: 4px 4px 0 0; border-radius: 999px; font-size: .72rem;
  background: rgba(139,92,246,.16); border: 1px solid rgba(139,92,246,.35); color: #d4c8ff;}
.cm-score {display: inline-block; margin-top: 6px; padding: 2px 9px; border-radius: 8px; font-size: .78rem; font-weight: 600;
  background: rgba(34,211,238,.12); border: 1px solid rgba(34,211,238,.4); color: #8be9f7;}
.cm-note {color: #a3a6bd; font-size: .85rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {transition: border-color .2s ease;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {border-color: rgba(139,92,246,.55);}
@media (max-width: 640px) {.cm-hero h1 {font-size: 1.7rem;} .block-container {padding-left: 1rem; padding-right: 1rem;}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- engine
@dataclass
class Engine:
    movies: pd.DataFrame
    recommender: MovieRecommender
    report: DataQualityReport
    data_dir: Path

    @property
    def ids(self) -> list[int]:
        return self.movies["movie_id"].tolist()

    def has(self, movie_id: object) -> bool:
        try:
            self.recommender.index_for_id(movie_id)  # type: ignore[arg-type]
            return True
        except Exception:
            return False

    def row(self, movie_id: int) -> pd.Series:
        return self.movies.iloc[self.recommender.index_for_id(movie_id)]

    def label_of(self, movie_id: int) -> str:
        return str(self.row(movie_id)["label"]) if self.has(movie_id) else str(movie_id)

    def frame_for(self, ids: list[int]) -> pd.DataFrame:
        """Rows for ``ids`` in the given order, silently skipping ids that are not in the dataset."""
        valid = [i for i in ids if self.has(i)]
        return self.movies.iloc[[self.recommender.index_for_id(i) for i in valid]].reset_index(drop=True)


@st.cache_resource(show_spinner="Preparing data and fitting the TF-IDF model (first run only)…")
def _build_engine(data_dir: str) -> Engine:
    movies_raw, credits_raw = load_raw_data(data_dir)
    movies, report = build_movie_table(movies_raw, credits_raw)
    return Engine(movies, MovieRecommender(movies), report, Path(data_dir))


def show_setup_message(err: DatasetNotFoundError) -> None:
    st.title("Dataset setup needed")
    st.error(f"CineMatch AI could not find: **{', '.join(err.missing)}**")
    st.markdown(
        f"""
The app needs the two TMDB 5000 files in this folder:

`{err.data_dir.resolve()}`

1. Download the **TMDB 5000 Movie Dataset** from Kaggle: [{KAGGLE_URL}]({KAGGLE_URL}) (free Kaggle account;
   please read the dataset's terms on that page).
2. Place **`{MOVIES_FILENAME}`** and **`{CREDITS_FILENAME}`** in the folder above
   (or set the `CINEMATCH_DATA_DIR` environment variable to another folder).
3. Reload this page.

No movie data is bundled with the app, and nothing is invented when the files are absent.
"""
    )


def get_engine() -> Engine:
    """Load (cached) data + model, or show a friendly message and stop the page."""
    try:
        return _build_engine(str(resolve_data_dir()))
    except DatasetNotFoundError as err:
        show_setup_message(err)
    except DatasetFormatError as err:
        st.error("The dataset files were found but could not be used.")
        st.write(str(err))
        st.caption("Re-download the original CSV files and try again.")
    except DatasetError as err:
        st.error(f"Dataset problem: {err}")
    except Exception as err:  # never show a raw traceback to end users
        st.error("Something went wrong while preparing the recommender.")
        with st.expander("Technical details"):
            st.code(f"{type(err).__name__}: {err}")
    st.stop()
    raise RuntimeError("unreachable")  # pragma: no cover


@st.cache_resource
def _store(path: str) -> LocalStore:
    return LocalStore(path)


def get_store() -> LocalStore:
    return _store(str(resolve_db_path()))


# ----------------------------------------------------------------------------- posters
def _secrets():
    try:
        return st.secrets
    except Exception:
        return None


def tmdb_api_key() -> str | None:
    return tmdb_client.get_api_key(secrets=_secrets())


@st.cache_resource
def _tmdb(key: str | None) -> tmdb_client.TMDBClient:
    return tmdb_client.TMDBClient(key)


def get_tmdb() -> tmdb_client.TMDBClient:
    return _tmdb(tmdb_api_key())


def poster_mode() -> str:
    client = get_tmdb()
    if not client.has_key:
        return "Dataset-only (no TMDB API key)"
    return "TMDB API (posters)" if client.enabled else f"Dataset-only ({client.disabled_reason})"


@lru_cache(maxsize=1)
def placeholder_uri() -> str:
    try:
        return "data:image/svg+xml;base64," + base64.b64encode(PLACEHOLDER.read_bytes()).decode()
    except OSError:
        return ""


def poster_urls(df: pd.DataFrame) -> dict[int, str | None]:
    """Dataset poster paths first; remaining ids via the optional TMDB API; else None."""
    urls: dict[int, str | None] = {}
    need: list[int] = []
    for movie_id, path in zip(df["movie_id"], df["poster_path"]):
        url = tmdb_client.poster_url(path)
        urls[int(movie_id)] = url
        if url is None:
            need.append(int(movie_id))
    client = get_tmdb()
    if need and client.enabled:
        urls.update(client.get_poster_urls(need))
    return urls


# --------------------------------------------------------------------- session helpers
def selected_movie_id() -> int | None:
    value = st.session_state.get("selected_movie_id")
    return int(value) if value is not None else None


def select_movie(movie_id: int) -> None:
    movie_id = int(movie_id)
    st.session_state["selected_movie_id"] = movie_id
    recent = [i for i in st.session_state.get("recent_ids", []) if i != movie_id]
    st.session_state["recent_ids"] = [movie_id] + recent[:7]


def go_details(movie_id: int) -> None:
    select_movie(movie_id)
    st.switch_page(PAGE_DETAILS)


def go_similar(movie_id: int) -> None:
    select_movie(movie_id)
    st.switch_page(PAGE_SIMILAR)


def movie_picker(engine: Engine, *, key: str, label: str = "Movie") -> int | None:
    """Searchable select box synced with the session's selected movie."""
    current = selected_movie_id()
    if current is not None and not engine.has(current):
        current = None
    ids = engine.ids
    index = ids.index(current) if current is not None else None
    choice = st.selectbox(
        label, options=ids, index=index, format_func=engine.label_of,
        placeholder="Type to search by title…", key=f"{key}_{current}",
        help="Click and start typing to filter the list. Movies sharing a title are shown with their release year.",
    )
    if choice is not None and choice != current:
        select_movie(choice)
    return choice


# ---------------------------------------------------------------------------- cards
def _na(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, str)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def fmt_rating(rating: object, votes: object) -> str:
    if _na(rating) or _na(votes) or int(votes) <= 0:
        return "Not rated"
    return f"★ {float(rating):.1f} · {int(votes):,} votes"


def fmt_runtime(minutes: object) -> str:
    if _na(minutes):
        return ""
    m = int(minutes)
    return f"{m // 60}h {m % 60:02d}m" if m >= 60 else f"{m}m"


def poster_html(url: str | None, title: str) -> str:
    src = html.escape(url or placeholder_uri(), quote=True)
    return f'<div class="cm-poster"><img src="{src}" alt="Poster of {html.escape(title, quote=True)}" loading="lazy"></div>'


def card_html(row: pd.Series, poster: str | None, score: float | None = None, extra: str = "") -> str:
    year = "" if _na(row["release_year"]) else str(int(row["release_year"]))
    chips = "".join(f'<span class="cm-chip">{html.escape(g)}</span>' for g in list(row["genres"])[:3])
    score_html = ""
    if score is not None:
        tip = html.escape(SIMILARITY_HELP, quote=True)
        score_html = f'<div><span class="cm-score" title="{tip}">Content similarity {score:.2f}</span></div>'
    meta = " · ".join(x for x in [year, fmt_rating(row["vote_average"], row["vote_count"])] if x)
    return (f'{poster_html(poster, row["title"])}'
            f'<div class="cm-title">{html.escape(str(row["title"]))}</div>'
            f'<div class="cm-meta">{html.escape(meta)}</div>{chips and f"<div>{chips}</div>"}{score_html}{extra}')


def render_grid(df: pd.DataFrame, *, key: str, n_cols: int = 5, show_score: bool = False,
                show_explanation: bool = False, remove_from: str | None = None) -> None:
    """Responsive grid of movie cards with working Details / Similar buttons.

    ``remove_from``: 'watchlist' | 'favorite' | 'rating' adds a working Remove button.
    """
    if df is None or df.empty:
        return
    posters = poster_urls(df)
    store = get_store() if remove_from else None
    records = df.reset_index(drop=True)
    for start in range(0, len(records), n_cols):
        cols = st.columns(n_cols)
        for offset, col in enumerate(cols):
            pos = start + offset
            if pos >= len(records):
                break
            row = records.iloc[pos]
            movie_id = int(row["movie_id"])
            score = float(row["similarity_score"]) if show_score and "similarity_score" in records.columns else None
            extra = ""
            if remove_from == "rating" and store is not None:
                mine = store.get_rating(movie_id)
                extra = f'<div class="cm-meta">Your rating: {"★" * (mine or 0)}{"☆" * (5 - (mine or 0))}</div>'
            with col, st.container(border=True):
                st.markdown(card_html(row, posters.get(movie_id), score, extra), unsafe_allow_html=True)
                if show_explanation and "explanation" in records.columns:
                    st.caption(f"**Why:** {row['explanation']}")
                if "mood_reason" in records.columns:
                    st.caption(f"**Matched on:** {row['mood_reason']}")
                b1, b2 = st.columns(2)
                if b1.button("Details", key=f"{key}_d_{pos}_{movie_id}", width="stretch"):
                    go_details(movie_id)
                if b2.button("Similar", key=f"{key}_s_{pos}_{movie_id}", width="stretch",
                             help="Find movies with similar content"):
                    go_similar(movie_id)
                if remove_from and store is not None:
                    if st.button("Remove", key=f"{key}_r_{pos}_{movie_id}", width="stretch"):
                        if remove_from == "rating":
                            store.clear_rating(movie_id)
                        else:
                            store.remove(movie_id, remove_from)
                        st.rerun()


def show_chart(fig) -> None:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
