"""Streamlit AppTest smoke tests: every page renders without an exception."""
import pytest
from streamlit.testing.v1 import AppTest

from pathlib import Path

from src.data_loader import DATA_DIR_ENV

from .conftest import requires_real_data

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["pages/1_Find_Similar_Movies.py", "pages/2_Movie_Explorer.py", "pages/3_Movie_Details.py",
         "pages/4_Mood_Discovery.py", "pages/5_Watchlist.py", "pages/6_ML_Insights.py", "pages/7_About.py"]


def test_missing_dataset_shows_setup_message_not_traceback(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("CINEMATCH_DB_PATH", str(tmp_path / "cm.db"))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    text = " ".join(e.value for e in app.error) + " ".join(m.value for m in app.markdown)
    assert "tmdb_5000_movies.csv" in text and "tmdb_5000_credits.csv" in text


@requires_real_data
def test_all_pages_render_and_core_flow_works(monkeypatch, tmp_path):
    monkeypatch.setenv("CINEMATCH_DB_PATH", str(tmp_path / "cm.db"))
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    for page in PAGES:
        app.switch_page(page).run()
        assert not app.exception, f"{page}: {[e.value for e in app.exception]}"

    # search -> Similar -> recommendations -> details -> watchlist
    app.switch_page("pages/0_Home.py").run()
    app.text_input(key="home_query").set_value("toy story").run()
    similar = [b for b in app.button if b.key and b.key.startswith("home_search_s_")]
    assert similar
    app.button(key=similar[0].key).click().run()
    assert not app.exception and app.session_state["selected_movie_id"] == 862  # Toy Story
    app.switch_page("pages/1_Find_Similar_Movies.py").run()  # settle the page (AppTest quirk after st.switch_page)
    assert [t.value for t in app.title] == ["Find Similar Movies"] and not app.exception
    details = [b for b in app.button if b.key and b.key.startswith("fs_grid_d_")]
    assert len(details) == 10
    app.button(key=details[0].key).click().run()
    movie_id = app.session_state["selected_movie_id"]
    assert movie_id != 862  # a recommendation, not the seed, was opened
    app.switch_page("pages/3_Movie_Details.py").run()
    app.button(key=f"wl_{movie_id}").click().run()
    app.switch_page("pages/5_Watchlist.py").run()
    assert app.tabs[0].label == "Watchlist (1)"
