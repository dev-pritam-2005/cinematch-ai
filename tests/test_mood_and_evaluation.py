import numpy as np
import pandas as pd
import pytest

from src import evaluation
from src.mood import MOODS, rank_by_mood, weighted_rating


def test_all_requested_moods_exist():
    assert {"Feel-good", "Adventure", "Suspense", "Thought-provoking", "Romantic", "Funny", "Relaxing"} <= set(MOODS)


def test_unknown_mood_raises(movie_table):
    with pytest.raises(KeyError):
        rank_by_mood(movie_table, "Nonexistent")


def test_romantic_mood_ranks_romance_first_and_explains(movie_table):
    out = rank_by_mood(movie_table, "Romantic", limit=5)
    assert set(out["title"].iloc[:2]) == {"Test Movie Wedding One", "Test Movie Wedding Two"}
    assert "Romance" in out["mood_reason"].iloc[0]
    assert (out["mood_score"] > 0).all()


def test_avoid_genres_and_filters(movie_table):
    heist = rank_by_mood(movie_table, "Feel-good")
    assert "Test Movie Heist" not in set(heist["title"])
    assert rank_by_mood(movie_table, "Romantic", min_votes=10_000).empty
    assert rank_by_mood(movie_table, "Romantic", year_range=(2005, 2020))["release_year"].between(2005, 2020).all()


def test_weighted_rating_shrinks_small_samples():
    votes, rating = pd.Series([1, 10_000]), pd.Series([10.0, 8.0])
    wr = weighted_rating(votes, rating, min_votes=100)
    assert abs(wr.iloc[0] - 9.0) < 0.1 < 10.0 - wr.iloc[0]  # 1-vote 10.0 is pulled to the mean (9.0)
    assert abs(wr.iloc[1] - 8.0) < 0.1  # 10,000 votes keep their own rating


# ---- evaluation
@pytest.mark.parametrize("title,stem", [("Toy Story 3", "toy story"), ("Star Trek: Nemesis", "star trek"),
                                        ("Friday the 13th Part VI: Jason Lives", "friday the 13th"),
                                        ("The Matrix Reloaded", "matrix reloaded"), ("Up", ""), (None, "")])
def test_franchise_stem(title, stem):
    assert evaluation.franchise_stem(title) == stem


def test_precision_recall_values():
    assert evaluation.precision_recall_at_k([1, 2, 3, 4], {2, 4, 9}, 2) == (0.5, 1 / 3)
    assert evaluation.precision_recall_at_k([1], set(), 3) == (0.0, 0.0)
    with pytest.raises(ValueError):
        evaluation.precision_recall_at_k([1], {1}, 0)


def test_franchise_proxy_on_fixture(recommender):
    summary, detail, info = evaluation.evaluate_franchise_proxy(recommender, ks=(1, 3))
    assert info["groups"] == 1 and info["queries"] == 2  # only the two "Same Title" movies share a stem
    assert summary["K"].tolist() == [1, 3]
    assert ((summary[["Precision@K", "Recall@K"]] >= 0) & (summary[["Precision@K", "Recall@K"]] <= 1)).all().all()


def test_coverage_and_diversity_on_fixture(recommender):
    result = evaluation.coverage_and_diversity(recommender, k=3, chunk=4)
    assert 0 < result["coverage"] <= 1
    assert result["top_scores"].shape == (len(recommender), 3)
    assert 0 <= result["mean_intra_list_diversity"] <= 1
    assert not np.isnan(result["median_topk_score"])
