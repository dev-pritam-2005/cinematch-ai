import streamlit as st

from src import ui
from src.recommender import AmbiguousTitleError, MovieNotFoundError

engine = ui.get_engine()
rec = engine.recommender

st.title("Find Similar Movies")
st.caption("Pick a movie and get the most similar titles by content: genres, keywords, plot, director and cast.")

movie_id = ui.movie_picker(engine, key="fs_picker", label="Choose a movie")

c1, c2, c3 = st.columns([1, 2, 2])
top_n = c1.radio("Results", [5, 10, 20], index=1, horizontal=True, key="fs_n")
genres = c2.multiselect("Only show genres", rec.all_genres, key="fs_genres",
                        help="Keeps recommendations that have at least one of the selected genres.")
bounds = rec.year_bounds
year_range = None
if bounds and bounds[0] < bounds[1]:
    chosen = c3.slider("Release years", bounds[0], bounds[1], bounds, key="fs_years")
    year_range = chosen if chosen != bounds else None

if movie_id is None:
    st.info("Select a movie above to see recommendations.", icon=":material/search:")
    st.stop()

seed = engine.row(movie_id)
st.markdown(f"### Because you chose **{seed['title']}**"
            + (f" ({int(seed['release_year'])})" if seed["release_year"] == seed["release_year"] else ""))
if not seed["overview"]:
    st.caption("This movie has no description in the dataset, so matches rely on genres, keywords and credits only.")
if not seed["genres"] and not seed["keywords"] and not seed["overview"]:
    st.warning("This movie has almost no usable metadata, so recommendations may be poor or empty.")

try:
    recs = rec.recommend(movie_id=movie_id, top_n=top_n, genres=genres or None, year_range=year_range)
except (MovieNotFoundError, AmbiguousTitleError) as err:
    st.error(str(err))
    st.stop()
except ValueError as err:
    st.error(f"Invalid request: {err}")
    st.stop()

if recs.empty:
    st.warning("No recommendations match these filters. Try removing a genre or widening the year range.")
else:
    if len(recs) < top_n:
        st.caption(f"Only {len(recs)} movie(s) matched the filters (you asked for {top_n}).")
    st.caption("Content similarity ranges from 0 to 1 and describes feature overlap, not how much you will enjoy a movie. "
               "Hover a score for details.")
    ui.render_grid(recs, key="fs_grid", n_cols=4, show_score=True, show_explanation=True)
