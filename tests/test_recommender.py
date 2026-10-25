import numpy as np
import pandas as pd
import pytest

from src.recommender import (AmbiguousTitleError, MovieNotFoundError, MovieRecommender, RecommenderError,
                             recommend_movies)

ID = {"space1": 1, "space2": 2, "wed1": 3, "wed2": 4, "heist": 5}


def test_most_similar_movie_ranks_first(recommender):
    recs = recommender.recommend("Test Movie Space One", top_n=3)
    assert recs["title"].iloc[0] == "Test Movie Space Two"
    assert recs["rank"].tolist() == [1, 2, 3][: len(recs)]


def test_selected_movie_is_never_recommended(recommender, movie_table):
    for mid in movie_table["movie_id"]:
        recs = recommender.recommend(movie_id=int(mid), top_n=20)
        assert int(mid) not in set(recs["movie_id"])


def test_results_are_distinct_sorted_and_bounded(recommender):
    recs = recommender.recommend(movie_id=ID["wed1"], top_n=20)
    assert recs["movie_id"].is_unique
    scores = recs["similarity_score"].to_numpy()
    assert (np.diff(scores) <= 1e-12).all()
    assert ((scores > 0) & (scores <= 1 + 1e-9)).all()


def test_top_n_larger_than_catalogue_returns_what_exists(recommender):
    recs = recommender.recommend("Test Movie Space One", top_n=500)
    assert 0 < len(recs) < len(recommender)


def test_unknown_title_raises_with_suggestion(recommender):
    with pytest.raises(MovieNotFoundError) as exc:
        recommender.recommend("Test Movie Spce One")
    assert "Test Movie Space One" in exc.value.suggestions


@pytest.mark.parametrize("bad", ["", "   ", "zzzzzz nothing", None])
def test_invalid_titles(recommender, bad):
    with pytest.raises((MovieNotFoundError, ValueError)):
        recommender.recommend(bad)


def test_unknown_movie_id_raises(recommender):
    with pytest.raises(MovieNotFoundError):
        recommender.recommend(movie_id=987654)


def test_ambiguous_title_lists_candidates_and_can_be_resolved(recommender):
    with pytest.raises(AmbiguousTitleError) as exc:
        recommender.recommend("same title")
    assert set(exc.value.candidates["movie_id"]) == {6, 7}
    assert recommender.recommend("Same Title (2001)", top_n=3).attrs["seed_movie_id"] == 6  # label disambiguates
    assert recommender.recommend(movie_id=7, top_n=3).attrs["seed_movie_id"] == 7


@pytest.mark.parametrize("bad_n", [0, -3, 2.5, "5", True, None])
def test_invalid_top_n(recommender, bad_n):
    with pytest.raises(ValueError):
        recommender.recommend("Test Movie Heist", top_n=bad_n)


def test_missing_selector_raises(recommender):
    with pytest.raises(ValueError):
        recommender.recommend()


def test_genre_filter(recommender):
    recs = recommender.recommend(movie_id=ID["space1"], top_n=10, genres=["action"])  # case-insensitive
    assert len(recs) and all("Action" in g for g in recs["genres"])
    assert recommender.recommend(movie_id=ID["space1"], genres=["Western"]).empty


def test_year_filter_and_unknown_years(recommender):
    recs = recommender.recommend(movie_id=ID["wed1"], top_n=10, year_range=(2000, 2012))
    assert recs["release_year"].between(2000, 2012).all()
    reversed_range = recommender.recommend(movie_id=ID["wed1"], top_n=10, year_range=(2012, 2000))
    assert set(reversed_range["movie_id"]) == set(recs["movie_id"])  # order of bounds is irrelevant
    assert recs["release_year"].notna().all()  # unknown-year movies excluded when filtering


def test_deterministic_tie_breaking(movie_table):
    twins = pd.concat([movie_table[movie_table["movie_id"] == 5]] * 3, ignore_index=True)
    twins["movie_id"] = [30, 10, 20]
    twins["title"] = ["Twin C", "Twin A", "Twin B"]
    twins["label"] = twins["title"]
    seed = movie_table[movie_table["movie_id"] == 5].assign(movie_id=1, title="Seed", label="Seed")
    rec = MovieRecommender(pd.concat([seed, twins], ignore_index=True), vectorizer_params={"min_df": 1})
    first = rec.recommend("Seed", top_n=3)
    assert first["movie_id"].tolist() == [10, 20, 30]  # equal scores -> ascending movie_id
    assert first.equals(rec.recommend("Seed", top_n=3))


def test_two_movie_catalogue_works(movie_table):
    rec = MovieRecommender(movie_table.iloc[:2].copy(), vectorizer_params={"min_df": 1})
    recs = rec.recommend(movie_id=1, top_n=5)
    assert recs["movie_id"].tolist() == [2]


def test_single_movie_catalogue_returns_empty(movie_table):
    rec = MovieRecommender(movie_table.iloc[:1].copy())
    assert rec.recommend(movie_id=1, top_n=5).empty


def test_empty_or_invalid_table_is_rejected(movie_table):
    with pytest.raises(RecommenderError):
        MovieRecommender(movie_table.iloc[0:0])
    with pytest.raises(RecommenderError):
        MovieRecommender(movie_table.drop(columns=["features"]))


def test_movie_with_no_overlap_gets_no_recommendations(recommender):
    assert recommender.recommend("Test Movie Heist", top_n=5).empty  # nothing shares a token with it


def test_featureless_movie_has_no_recommendations_but_no_crash(recommender):
    assert recommender.recommend(movie_id=8, top_n=5).empty  # movie 8: no genres/overview/credits


def test_min_df_falls_back_on_tiny_data(movie_table):
    disjoint = movie_table[movie_table["movie_id"].isin([1, 5])].copy()  # share no tokens -> min_df=2 leaves nothing
    rec = MovieRecommender(disjoint)
    assert rec.matrix.shape[1] > 0 and rec.params["min_df"] == 1


def test_explanations_only_claim_verifiable_overlap(recommender, movie_table):
    recs = recommender.recommend(movie_id=ID["space1"], top_n=10)
    seed = movie_table[movie_table["movie_id"] == 1].iloc[0]
    for _, r in recs.iterrows():
        exp = r["explanation"]
        shared_genres = set(seed["genres"]) & set(r["genres"])
        assert ("Shared genres" in exp) == bool(shared_genres)
        assert ("Same director" in exp) == (seed["director"] == r["director"])
        assert ("Shared cast" in exp) == bool(set(seed["cast"]) & set(r["cast"]))
        assert ("Shared keywords" in exp) == bool(set(seed["keywords"]) & set(r["keywords"]))
    top = recs.iloc[0]
    assert "Same director: Dana Director" in top["explanation"] and "Ann Actor" in top["explanation"]


def test_search(recommender):
    assert recommender.search("").empty and recommender.search("qqqqq").empty
    hits = recommender.search("test movie w")
    assert set(hits["title"]) == {"Test Movie Wedding One", "Test Movie Wedding Two"}
    assert recommender.search("heist")["title"].iloc[0] == "Test Movie Heist"
    assert len(recommender.search("test", limit=2)) == 2 and recommender.search("test", limit=0).empty


def test_module_level_function(recommender):
    recs = recommend_movies("Test Movie Wedding One", 2, recommender=recommender)
    assert recs["title"].iloc[0] == "Test Movie Wedding Two"
    assert {"similarity_score", "explanation", "genres"} <= set(recs.columns)


def test_all_genres_and_year_bounds(recommender):
    assert "Romance" in recommender.all_genres
    assert recommender.year_bounds == (1999, 2015)
