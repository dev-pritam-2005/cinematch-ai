"""Shared fixtures.

IMPORTANT: the movies below are SYNTHETIC test fixtures ("Test Movie ...") used only by the
unit tests. They are never shown in the app and are not real films.
"""
import json
import os

import pandas as pd
import pytest

from src.data_loader import CREDITS_FILENAME, MOVIES_FILENAME, resolve_data_dir
from src.preprocessing import build_movie_table
from src.recommender import MovieRecommender


def _names(names):
    return json.dumps([{"id": i, "name": n} for i, n in enumerate(names)])


def _cast(names):
    return json.dumps([{"cast_id": i, "name": n, "order": i} for i, n in enumerate(names)])


def _crew(director, extra=("Some Editor",)):
    people = [{"name": e, "job": "Editor"} for e in extra]
    if director:
        people.append({"name": director, "job": "Director"})
    return json.dumps(people)


def _movie(i, title, genres, kws, overview, date, votes=100, avg=7.0, runtime=100.0):
    return {"id": i, "title": title, "original_title": title, "overview": overview, "genres": _names(genres),
            "keywords": _names(kws), "release_date": date, "vote_average": avg, "vote_count": votes,
            "runtime": runtime, "original_language": "en", "popularity": float(i)}


@pytest.fixture
def raw_frames():
    movies = [
        _movie(1, "Test Movie Space One", ["Science Fiction", "Action"], ["space war", "alien invasion"],
               "A crew of astronauts fights an alien invasion in deep space.", "2001-05-01"),
        _movie(2, "Test Movie Space Two", ["Science Fiction", "Action"], ["space war", "alien invasion"],
               "Astronauts battle aliens again across the galaxy.", "2004-06-01"),
        _movie(3, "Test Movie Wedding One", ["Romance", "Comedy"], ["wedding", "small town"],
               "Two strangers fall in love at a wedding in a small town.", "1999-02-14"),
        _movie(4, "Test Movie Wedding Two", ["Romance", "Comedy"], ["wedding", "love"],
               "A wedding planner falls in love with the groom's brother.", "2010-02-14"),
        _movie(5, "Test Movie Heist", ["Crime", "Thriller"], ["bank robbery", "heist"],
               "A gang plans an elaborate bank robbery.", "2008-09-09"),
        _movie(6, "Same Title", ["Drama"], ["family"], "A quiet family drama about two sisters.", "2001-01-01", votes=50),
        _movie(7, "Same Title", ["Comedy"], ["road trip"], "Friends take a chaotic road trip.", "2010-01-01", votes=60),
        _movie(8, "Test Movie No Overview", [], [], "", "2012-12-12", votes=0, avg=0.0, runtime=0.0),
        _movie(9, "Test Movie Bad Date", ["Drama"], ["family"], "A family drama set in winter.", "not-a-date"),
    ]
    df = pd.DataFrame(movies)
    df.loc[len(df)] = df.iloc[0]  # exact duplicate id
    df.loc[len(df)] = {**_movie(10, "Test Movie Broken Json", ["Drama"], [], "Broken metadata.", "2015-05-05"),
                       "genres": "[{'id': 1, 'name': 'Drama'"}  # malformed JSON
    credits = pd.DataFrame({
        "movie_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "title": [m["title"] for m in movies],
        "cast": [_cast(["Ann Actor", "Bo Star", "Cy Extra", "Di Late"]), _cast(["Ann Actor", "Eve Lead"]),
                 _cast(["Fay Romance"]), _cast(["Fay Romance", "Gus Groom"]), _cast(["Hal Thief"]),
                 _cast(["Ida Sister"]), _cast(["Jo Driver"]), "[]", _cast(["Kai Winter"])],
        "crew": [_crew("Dana Director"), _crew("Dana Director"), _crew("Eli Maker"), _crew("Eli Maker"),
                 _crew("Finn Caper"), _crew(None), _crew("Gwen Road"), "[]", _crew("Hugh Cold")],
    })  # movie 10 intentionally has no credits row
    return df, credits


@pytest.fixture
def movie_table(raw_frames):
    table, report = build_movie_table(*raw_frames)
    return table


@pytest.fixture
def recommender(movie_table):
    return MovieRecommender(movie_table, vectorizer_params={"min_df": 1})


@pytest.fixture
def csv_dir(tmp_path, raw_frames):
    movies, credits = raw_frames
    movies.to_csv(tmp_path / MOVIES_FILENAME, index=False)
    credits.to_csv(tmp_path / CREDITS_FILENAME, index=False)
    return tmp_path


def real_dataset_dir():
    """Directory holding the real TMDB CSVs, or None."""
    base = resolve_data_dir()
    ok = (base / MOVIES_FILENAME).is_file() and (base / CREDITS_FILENAME).is_file()
    return base if ok else None


requires_real_data = pytest.mark.skipif(real_dataset_dir() is None,
                                        reason="TMDB CSVs not found (see data/README.md or set CINEMATCH_DATA_DIR)")
