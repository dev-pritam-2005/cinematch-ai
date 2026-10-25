import streamlit as st

from src import ui
from src.mood import AVOID_PENALTY, KEYWORD_CAP, KEYWORD_POINTS, MOODS, TEXT_CAP, TEXT_POINTS, rank_by_mood

engine = ui.get_engine()
rec = engine.recommender

st.title("Mood-Based Discovery")
st.info("Mood matches are **approximate**. They come from simple, visible rules over each movie's genres, keywords and "
        "description — not from a trained emotion model — and depend on the dataset's metadata.",
        icon=":material/info:")

mood = st.pills("What are you in the mood for?", list(MOODS), selection_mode="single", default="Feel-good", key="mood_pick")
if not mood:
    st.caption("Choose a mood above.")
    st.stop()
st.caption(MOODS[mood].blurb)

bounds = rec.year_bounds
c1, c2, c3 = st.columns(3)
min_votes = c1.select_slider("Minimum votes", [0, 50, 100, 200, 500, 1000], value=100, key="mood_votes",
                             help="Hides movies with very few ratings.")
year_range = None
if bounds and bounds[0] < bounds[1]:
    chosen = c2.slider("Release years", bounds[0], bounds[1], bounds, key="mood_years")
    year_range = chosen if chosen != bounds else None
limit = c3.selectbox("Results", [10, 20, 40], key="mood_limit")

results = rank_by_mood(engine.movies, mood, min_votes=min_votes, year_range=year_range, limit=limit)
with st.expander("How is this mood matched?"):
    rule = MOODS[mood]
    st.markdown(
        f"- **Genres (points):** {', '.join(f'{g} {w:g}' for g, w in rule.genre_weights.items())}\n"
        f"- **Keyword phrases containing:** {', '.join(rule.keywords)} ({KEYWORD_POINTS:g} point each, max {KEYWORD_CAP})\n"
        f"- **Description words:** {', '.join(rule.overview_terms)} ({TEXT_POINTS:g} point each, max {TEXT_CAP})\n"
        f"- **Penalty:** −{AVOID_PENALTY:g} if the movie has any of: {', '.join(rule.avoid_genres) or 'none'}\n"
        "- A movie must match at least one genre or keyword. Ties are ordered by a vote-weighted average rating."
    )

if results.empty:
    st.warning("No movies match this mood with the current filters. Lower the minimum votes or widen the years.")
else:
    st.caption(f"Top {len(results)} matches. “Matched on” lists the real fields that triggered each match.")
    ui.render_grid(results, key="mood_grid", n_cols=4)
