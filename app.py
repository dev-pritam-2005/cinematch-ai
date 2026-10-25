"""CineMatch AI - entry point. Run with:  streamlit run app.py"""
import streamlit as st

try:  # optional: load TMDB_API_KEY etc. from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from src import tmdb_client, ui

st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")
ui.inject_css()

pages = [
    st.Page(ui.PAGE_HOME, title="Discover", icon=":material/home:", default=True),
    st.Page(ui.PAGE_SIMILAR, title="Find Similar Movies", icon=":material/search:"),
    st.Page(ui.PAGE_EXPLORER, title="Movie Explorer", icon=":material/grid_view:"),
    st.Page(ui.PAGE_DETAILS, title="Movie Details", icon=":material/movie:"),
    st.Page(ui.PAGE_MOOD, title="Mood Discovery", icon=":material/mood:"),
    st.Page(ui.PAGE_WATCHLIST, title="Watchlist & Favorites", icon=":material/bookmark:"),
    st.Page(ui.PAGE_INSIGHTS, title="ML Insights", icon=":material/insights:"),
    st.Page(ui.PAGE_ABOUT, title="About", icon=":material/info:"),
]
navigation = st.navigation(pages)

with st.sidebar:
    st.caption(f"Poster mode: {ui.poster_mode()}")
    if ui.get_tmdb().enabled:
        st.caption(tmdb_client.ATTRIBUTION)

navigation.run()
