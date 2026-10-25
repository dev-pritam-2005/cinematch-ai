"""Integration tests on the real TMDB 5000 files (skipped automatically when absent)."""
import pytest

from src.data_loader import load_raw_data
from src.preprocessing import build_movie_table
from src.recommender import AmbiguousTitleError, MovieNotFoundError, MovieRecommender

from .conftest import requires_real_data

pytestmark = requires_real_data


@pytest.fixture(scope="module")
def real():
    movies, report = build_movie_table(*load_raw_data())
    return movies, report, MovieRecommender(movies)


def test_dataset_loads_and_is_clean(real):
    movies, report, _ = real
    assert len(movies) > 4000 and movies["movie_id"].is_unique and movies["label"].is_unique
    assert sum(report.malformed_json.values()) == 0
    assert (movies["vote_average"].isna() == (movies["vote_count"] == 0)).all()  # unrated <=> zero votes
    assert not movies["features"].eq("").all()


def test_recommendations_are_sensible(real):
    _, _, rec = real
    dark = rec.recommend("The Dark Knight", top_n=5)
    assert "The Dark Knight Rises" in set(dark["title"]) and "Batman Begins" in set(dark["title"])
    toy = rec.recommend("Toy Story", top_n=3)
    assert {"Toy Story 2", "Toy Story 3"} <= set(toy["title"])
    for recs in (dark, toy):
        assert recs["movie_id"].is_unique and recs["similarity_score"].between(0, 1).all()


def test_selected_movie_excluded_for_every_movie_sample(real):
    movies, _, rec = real
    for mid in movies["movie_id"].sample(60, random_state=0):
        assert int(mid) not in set(rec.recommend(movie_id=int(mid), top_n=10, explain=False)["movie_id"])


def test_real_ambiguous_and_missing_titles(real):
    _, _, rec = real
    with pytest.raises(AmbiguousTitleError):
        rec.recommend("Batman")
    assert rec.recommend("Batman (1989)", top_n=3).attrs["seed_movie_id"] == 268
    with pytest.raises(MovieNotFoundError):
        rec.recommend("Definitely Not A Real Movie 12345")
