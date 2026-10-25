import numpy as np
import pandas as pd

from src.preprocessing import (FeatureWeights, build_feature_text, build_movie_table, extract_cast, extract_director,
                               extract_names, language_name, safe_json_loads, to_token)


# ---- JSON parsing
def test_safe_json_loads_variants():
    assert safe_json_loads('[{"name": "A"}]') == ([{"name": "A"}], "ok")
    assert safe_json_loads("[{'name': 'A'}]")[1] == "ok"  # python-literal fallback
    assert safe_json_loads('{"name": "A"}') == ([{"name": "A"}], "ok")  # single dict wrapped
    assert safe_json_loads(None) == ([], "missing")
    assert safe_json_loads(float("nan")) == ([], "missing")
    assert safe_json_loads("   ") == ([], "missing")
    assert safe_json_loads("[{broken") == ([], "malformed")
    assert safe_json_loads("42") == ([], "malformed")
    assert safe_json_loads(123) == ([], "malformed")
    assert safe_json_loads([{"name": "x"}])[1] == "ok"


def test_extract_names_skips_bad_entries_and_dedupes():
    items = [{"name": "Drama"}, {"name": " drama "}, {"id": 1}, "junk", None, {"name": 5}, {"name": "Comedy"}]
    assert extract_names(items) == ["Drama", "Comedy"]
    assert extract_names(items, limit=1) == ["Drama"]


def test_extract_cast_orders_and_limits():
    items = [{"name": "C", "order": 2}, {"name": "A", "order": 0}, {"name": "B", "order": 1}, {"name": "Z"}]
    assert extract_cast(items, 3) == ["A", "B", "C"]
    assert extract_cast(items, 0) == []
    assert extract_cast([{"name": "x", "order": float("nan")}], 5) == ["x"]


def test_extract_director():
    crew = [{"job": "Editor", "name": "E"}, {"job": "Director", "name": "  Jane   Doe "}]
    assert extract_director(crew) == "Jane Doe"
    assert extract_director([{"job": "Editor", "name": "E"}]) == ""
    assert extract_director([]) == ""


def test_to_token_and_language():
    assert to_token("Science Fiction") == "science_fiction"
    assert to_token("Sam  Worthington") == "sam_worthington"
    assert to_token("") == ""
    assert language_name("en") == "English" and language_name("xx") == "XX" and language_name("") == ""


# ---- feature preparation
def test_feature_weights_repeat_tokens():
    text = build_feature_text(genres=["Drama"], keywords=["road trip"], director="Jo Lee", cast=["Al Pa"],
                              overview="A story", weights=FeatureWeights(genres=3, keywords=2, director=2, cast=1, overview=1))
    tokens = text.split()
    assert tokens.count("drama") == 3 and tokens.count("road") == 2 and tokens.count("jo_lee") == 2
    assert tokens.count("al_pa") == 1 and tokens.count("story") == 1


def test_feature_text_handles_empty_inputs():
    assert build_feature_text(genres=[], keywords=[], director="", cast=[], overview="") == ""


# ---- merged table
def test_table_basics(raw_frames):
    table, report = build_movie_table(*raw_frames)
    assert report.duplicate_ids_removed == 1
    assert len(table) == 10 and table["movie_id"].is_unique
    assert report.malformed_json["genres"] == 1
    assert report.rows_without_credits == 1  # movie 10 has no credits row but is kept
    assert report.invalid_release_dates == 1
    assert report.missing_overview == 1


def test_metadata_is_kept_separately_from_features(movie_table):
    row = movie_table[movie_table["movie_id"] == 1].iloc[0]
    assert row["genres"] == ["Science Fiction", "Action"]  # original names, not tokens
    assert row["director"] == "Dana Director"
    assert row["cast"] == ["Ann Actor", "Bo Star", "Cy Extra"]  # default cast_size = 3
    assert "science_fiction" in row["features"] and "Science Fiction" not in row["features"]


def test_cast_size_is_configurable(raw_frames):
    table, _ = build_movie_table(*raw_frames, cast_size=1)
    assert table[table["movie_id"] == 1].iloc[0]["cast"] == ["Ann Actor"]


def test_missing_values_are_handled_not_invented(movie_table):
    blank = movie_table[movie_table["movie_id"] == 8].iloc[0]
    assert blank["overview"] == "" and blank["genres"] == [] and blank["director"] == ""
    assert np.isnan(blank["vote_average"])  # 0 votes -> unrated, not "rated 0"
    assert np.isnan(blank["runtime"])  # runtime 0 -> unknown
    assert blank["poster_path"] is None  # never fabricated


def test_invalid_date_becomes_missing_year(movie_table):
    bad = movie_table[movie_table["movie_id"] == 9].iloc[0]
    assert pd.isna(bad["release_date"]) and pd.isna(bad["release_year"])
    assert bad["label"] == "Test Movie Bad Date"


def test_malformed_json_row_survives_with_empty_genres(movie_table):
    row = movie_table[movie_table["movie_id"] == 10].iloc[0]
    assert row["genres"] == [] and row["cast"] == [] and row["director"] == ""


def test_duplicate_titles_get_unique_labels(movie_table):
    same = movie_table[movie_table["title"] == "Same Title"]
    assert len(same) == 2
    assert set(same["label"]) == {"Same Title (2001)", "Same Title (2010)"}
    assert movie_table["label"].is_unique


def test_same_title_and_date_duplicates_collapse(raw_frames):
    movies, credits = raw_frames
    twin = movies[movies["id"] == 3].copy()
    twin["id"], twin["vote_count"] = 99, 1
    table, report = build_movie_table(pd.concat([movies, twin], ignore_index=True), credits)
    assert report.duplicate_title_date_removed == 1
    assert 99 not in set(table["movie_id"])  # the lower-vote record is dropped


def test_rows_with_bad_ids_are_dropped(raw_frames):
    movies, credits = raw_frames
    movies = movies.copy()
    movies["id"] = movies["id"].astype(object)
    movies.loc[4, "id"] = "oops"  # movie 5 has no duplicate row
    table, report = build_movie_table(movies, credits)
    assert report.rows_dropped_bad_id == 1 and 5 not in set(table["movie_id"])
