import math

import pandas as pd
import streamlit as st

from src import ui

engine = ui.get_engine()
movies, rec = engine.movies, engine.recommender
bounds = rec.year_bounds
SORTS = {
    "Rating (high → low)": (["vote_average", "_t", "movie_id"], [False, True, True]),
    "Release year (newest first)": (["release_year", "_t", "movie_id"], [False, True, True]),
    "Release year (oldest first)": (["release_year", "_t", "movie_id"], [True, True, True]),
    "Title (A → Z)": (["_t", "movie_id"], [True, True]),
}
DEFAULTS = {"ex_q": "", "ex_genres": [], "ex_all": False, "ex_years": bounds, "ex_rating": 0.0,
            "ex_votes": 10, "ex_sort": list(SORTS)[0], "ex_size": 20, "ex_page": 1}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_filters() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


st.title("Movie Explorer")
st.caption("Filter and sort the real dataset. Ratings are TMDB community averages; a minimum vote count avoids "
           "over-trusting movies with only a handful of votes.")

c1, c2 = st.columns([2, 3])
query = c1.text_input("Title contains", key="ex_q", placeholder="e.g. star")
genres = c2.multiselect("Genres", rec.all_genres, key="ex_genres")
c3, c4, c5, c6 = st.columns(4)
match_all = c3.toggle("Match all genres", key="ex_all", help="Off: movie has at least one selected genre.")
years = c4.slider("Release years", bounds[0], bounds[1], key="ex_years") if bounds and bounds[0] < bounds[1] else None
min_rating = c5.slider("Minimum rating", 0.0, 10.0, step=0.5, key="ex_rating")
min_votes = c6.select_slider("Minimum votes", [0, 10, 50, 100, 500, 1000, 5000], key="ex_votes")
c7, c8, c9 = st.columns([3, 1, 1])
sort_by = c7.selectbox("Sort by", list(SORTS), key="ex_sort")
page_size = c8.selectbox("Per page", [10, 20, 40], key="ex_size")
c9.button("Reset filters", on_click=reset_filters, width="stretch")

mask = pd.Series(True, index=movies.index)
if query.strip():
    mask &= movies["title"].str.casefold().str.contains(query.strip().casefold(), regex=False)
if genres:
    wanted = set(genres)
    mask &= movies["genres"].map((lambda gs: wanted <= set(gs)) if match_all else (lambda gs: bool(wanted & set(gs))))
if years is not None and tuple(years) != bounds:
    mask &= movies["release_year"].between(years[0], years[1]).fillna(False).astype(bool)
if min_rating > 0:
    mask &= (movies["vote_average"] >= min_rating).fillna(False).astype(bool)
mask &= movies["vote_count"] >= min_votes

view = movies[mask].assign(_t=lambda d: d["title"].str.casefold())
cols, asc = SORTS[sort_by]
view = view.sort_values(cols, ascending=asc, kind="stable", na_position="last")

total = len(view)
pages_total = max(1, math.ceil(total / page_size))
signature = (query, tuple(genres), match_all, tuple(years or ()), min_rating, min_votes, sort_by, page_size)
if st.session_state.get("ex_sig") != signature:
    st.session_state["ex_sig"] = signature
    st.session_state["ex_page"] = 1
st.session_state["ex_page"] = int(min(max(1, st.session_state["ex_page"]), pages_total))

if total == 0:
    st.warning("No movies match these filters. Try lowering the minimum rating/votes or removing a genre.")
    st.stop()

page = st.number_input("Page", min_value=1, max_value=pages_total, step=1, key="ex_page") if pages_total > 1 else 1
first = (page - 1) * page_size
st.caption(f"Showing {first + 1}–{min(first + page_size, total)} of {total:,} matching movies (page {page} of {pages_total}).")
ui.render_grid(view.iloc[first:first + page_size].drop(columns="_t"), key=f"ex_grid_{page}", n_cols=5)
