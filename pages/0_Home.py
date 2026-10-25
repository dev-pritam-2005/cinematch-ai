import streamlit as st

from src import ui

engine = ui.get_engine()
movies = engine.movies

st.markdown(
    f"""<div class="cm-hero"><h1><span class="cm-accent">CineMatch AI</span></h1>
<p>Find movies similar to the ones you love. Recommendations are content-based: each of the {len(movies):,} movies
is compared by genres, keywords, plot description, director and top-billed cast using TF-IDF and cosine similarity.</p></div>""",
    unsafe_allow_html=True,
)

query = st.text_input("Search for a movie", placeholder="Try “Inception”, “Toy Story” or “Alien”", key="home_query",
                      help="Searches movie titles in the local dataset.")
if query.strip():
    found = engine.recommender.search(query, limit=8)
    if found.empty:
        st.info(f"No movie in the dataset matches “{query.strip()}”. Check the spelling or try a shorter title.")
    else:
        st.caption(f"{len(found)} match(es) — use **Similar** to find movies like one of them.")
        ui.render_grid(found, key="home_search", n_cols=4)

if st.button("Discover similar movies", type="primary", icon=":material/arrow_forward:"):
    st.switch_page(ui.PAGE_SIMILAR)

recent = engine.frame_for(st.session_state.get("recent_ids", []))
if not recent.empty:
    st.subheader("Recently selected")
    st.caption("Movies you opened or searched during this session.")
    ui.render_grid(recent.head(5), key="home_recent")
    seed = recent.iloc[0]
    st.subheader(f"Because you looked at {seed['title']}")
    recs = engine.recommender.recommend(movie_id=int(seed["movie_id"]), top_n=5)
    ui.render_grid(recs, key="home_because", show_score=True)

st.subheader("Highest TMDB popularity score in this dataset")
st.caption("Ranked by the `popularity` value stored in the dataset file, a snapshot from when it was created — not live trending data.")
featured = movies.dropna(subset=["popularity"]).sort_values(["popularity", "movie_id"], ascending=[False, True]).head(10)
ui.render_grid(featured, key="home_featured", n_cols=5)

st.subheader("Explore by genre")
counts = movies.explode("genres")["genres"].value_counts()
genre = st.pills("Genre", options=counts.index.tolist(), selection_mode="single", key="home_genre",
                 label_visibility="collapsed")
if genre:
    in_genre = movies[movies["genres"].map(lambda gs: genre in gs) & (movies["vote_count"] >= 200)]
    top = in_genre.sort_values(["vote_average", "vote_count", "movie_id"], ascending=[False, False, True]).head(10)
    st.caption(f"Top-rated {genre} movies with at least 200 votes ({len(in_genre):,} qualify).")
    ui.render_grid(top, key="home_genre_grid", n_cols=5)
else:
    st.caption("Pick a genre to see its highest-rated movies (minimum 200 votes).")
