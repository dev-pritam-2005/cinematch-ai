import html

import streamlit as st

from src import ui
from src.preprocessing import language_name

engine = ui.get_engine()
store = ui.get_store()

st.title("Movie Details")
movie_id = ui.movie_picker(engine, key="details_picker", label="Choose a movie")
if movie_id is None:
    st.info("Search for a movie above, or open one with the **Details** button on any other page.",
            icon=":material/movie:")
    st.stop()

row = engine.row(movie_id)
poster = ui.poster_urls(engine.frame_for([movie_id])).get(movie_id)
year = "" if row["release_year"] != row["release_year"] else f" ({int(row['release_year'])})"

left, right = st.columns([1, 2.4], gap="large")
with left:
    st.markdown(ui.poster_html(poster, row["title"]), unsafe_allow_html=True)
    if poster is None:
        st.caption("No poster in the dataset. Add a TMDB API key to load posters (see About).")

with right:
    st.header(f"{row['title']}{year}")
    facts = [ui.fmt_rating(row["vote_average"], row["vote_count"]), ui.fmt_runtime(row["runtime"]),
             language_name(row["original_language"])]
    st.markdown(" · ".join(f for f in facts if f))
    if row["genres"]:
        st.markdown("".join(f'<span class="cm-chip">{html.escape(g)}</span>' for g in row["genres"]),
                    unsafe_allow_html=True)
    st.markdown("#### Overview")
    st.write(row["overview"] or "No description is available for this movie in the dataset.")
    st.markdown(f"**Director:** {row['director'] or 'Not listed'}")
    st.markdown(f"**Top cast:** {', '.join(row['cast']) if row['cast'] else 'Not listed'}")
    if row["keywords"]:
        with st.expander(f"Keywords ({len(row['keywords'])})"):
            st.write(", ".join(row["keywords"]))

    b1, b2, b3 = st.columns(3)
    on_list = store.contains(movie_id, "watchlist")
    if b1.button("Remove from watchlist" if on_list else "Add to watchlist", icon=":material/bookmark:",
                 type="secondary" if on_list else "primary", width="stretch", key=f"wl_{movie_id}"):
        store.toggle(movie_id, "watchlist")
        st.rerun()
    is_fav = store.contains(movie_id, "favorite")
    if b2.button("Remove favorite" if is_fav else "Mark as favorite", icon=":material/favorite:",
                 width="stretch", key=f"fav_{movie_id}"):
        store.toggle(movie_id, "favorite")
        st.rerun()

    mine = store.get_rating(movie_id)
    nonce = st.session_state.get(f"fb_nonce_{movie_id}", 0)
    st.caption("Your rating (saved locally)")
    stars = st.feedback("stars", key=f"fb_{movie_id}_{nonce}", default=(mine - 1) if mine else None)
    if stars is not None and stars + 1 != mine:
        store.set_rating(movie_id, stars + 1)
        st.toast(f"Saved your rating: {stars + 1}/5")
        mine = stars + 1
    if mine and st.button("Clear my rating", key=f"clr_{movie_id}"):
        store.clear_rating(movie_id)
        st.session_state[f"fb_nonce_{movie_id}"] = nonce + 1
        st.rerun()

st.divider()
st.subheader("Similar movies")
n = st.radio("How many", [5, 10], horizontal=True, key="details_n")
recs = engine.recommender.recommend(movie_id=movie_id, top_n=n)
if recs.empty:
    st.info("No similar movies could be found from this movie's available metadata.")
else:
    st.caption("Content similarity is feature overlap, not a prediction of your enjoyment.")
    ui.render_grid(recs, key="details_grid", n_cols=5, show_score=True, show_explanation=True)
