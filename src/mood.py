"""Transparent, rule-based mood matching (NOT a trained emotion model).

Each mood is a small, readable rule set over three real dataset fields:
genres (weighted), keyword phrases (substring match) and overview words.
score = genre weights + 1.0 per matching keyword (max 3) + 0.5 per matching overview
word (max 3) - 3.0 if the movie has an "avoid" genre. A movie must match at least one
genre weight or keyword to qualify. Ties are broken by a Bayesian weighted rating.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocessing import normalize_words

KEYWORD_POINTS, KEYWORD_CAP = 1.0, 3
TEXT_POINTS, TEXT_CAP = 0.5, 3
AVOID_PENALTY = 3.0


@dataclass(frozen=True)
class MoodRule:
    name: str
    blurb: str
    genre_weights: dict[str, float]
    keywords: tuple[str, ...]
    overview_terms: tuple[str, ...]
    avoid_genres: tuple[str, ...] = ()


MOODS: dict[str, MoodRule] = {m.name: m for m in (
    MoodRule("Feel-good", "Warm, upbeat stories.",
             {"Comedy": 2, "Family": 2, "Animation": 1, "Romance": 1, "Music": 1.5, "Adventure": 0.5},
             ("friendship", "family", "feel good", "wedding", "happy", "christmas", "dream", "musical", "coming of age"),
             ("friendship", "hope", "dream", "family", "joy", "heartwarming", "together"),
             ("Horror", "War", "Crime", "Thriller")),
    MoodRule("Adventure", "Quests, journeys and big set pieces.",
             {"Adventure": 3, "Action": 1.5, "Fantasy": 1.5, "Science Fiction": 1},
             ("quest", "journey", "treasure", "expedition", "island", "pirate", "survival", "road trip", "explorer"),
             ("quest", "journey", "adventure", "treasure", "expedition", "escape", "voyage"),
             ("Documentary",)),
    MoodRule("Suspense", "Tension, mystery and edge-of-the-seat plots.",
             {"Thriller": 3, "Mystery": 2.5, "Crime": 1.5, "Horror": 1},
             ("suspense", "serial killer", "kidnapping", "conspiracy", "investigation", "detective", "murder", "hostage", "psychopath"),
             ("murder", "killer", "mystery", "detective", "investigation", "secret", "kidnapped"),
             ("Family", "Animation", "Comedy")),
    MoodRule("Thought-provoking", "Ideas, moral questions and ambiguity.",
             {"Drama": 1, "Science Fiction": 1.5, "Mystery": 1, "History": 1.5, "Documentary": 2},
             ("philosophy", "dystopia", "identity", "artificial intelligence", "time travel", "memory", "social commentary",
              "based on true story", "moral dilemma", "ethics", "consciousness", "alternate reality"),
             ("identity", "reality", "memory", "society", "truth", "consciousness", "moral"),
             ("Family", "Animation")),
    MoodRule("Romantic", "Love stories and relationships.",
             {"Romance": 3, "Drama": 0.5, "Comedy": 0.5},
             ("love", "romance", "wedding", "marriage", "first love", "love triangle", "relationship", "falling in love", "couple"),
             ("love", "romance", "relationship", "marriage", "wedding", "heart"),
             ("Horror", "War")),
    MoodRule("Funny", "Comedies built to make you laugh.",
             {"Comedy": 3, "Animation": 0.5, "Family": 0.5},
             ("parody", "satire", "slapstick", "spoof", "farce", "comedy", "humor", "buddy", "dark comedy"),
             ("hilarious", "comic", "funny", "misadventures", "prank", "laugh"),
             ("Horror", "War", "Documentary")),
    MoodRule("Relaxing", "Gentle, low-stress viewing.",
             {"Animation": 1.5, "Family": 2, "Music": 1.5, "Romance": 1, "Comedy": 1, "Fantasy": 0.5},
             ("small town", "nature", "friendship", "cooking", "holiday", "vacation", "village", "countryside", "travel", "gentle"),
             ("peaceful", "gentle", "village", "summer", "holiday", "nature", "charming"),
             ("Horror", "Thriller", "War", "Crime", "Action")),
)}


def weighted_rating(votes: pd.Series, rating: pd.Series, min_votes: float = 100.0) -> pd.Series:
    """IMDB-style Bayesian average: shrinks ratings with few votes toward the global mean."""
    valid = rating.dropna()
    mean = float(valid.mean()) if len(valid) else 0.0
    v = votes.astype(float)
    r = rating.fillna(mean)
    return (v / (v + min_votes)) * r + (min_votes / (v + min_votes)) * mean


def rank_by_mood(
    movies: pd.DataFrame,
    mood: str,
    *,
    min_votes: int = 0,
    year_range: tuple[int, int] | None = None,
    limit: int = 24,
) -> pd.DataFrame:
    """Rank movies for ``mood``; returns columns ``mood_score`` and ``mood_reason``."""
    if mood not in MOODS:
        raise KeyError(f"Unknown mood {mood!r}. Choose one of: {', '.join(MOODS)}")
    rule = MOODS[mood]
    weights = {k.casefold(): v for k, v in rule.genre_weights.items()}
    avoid = {a.casefold() for a in rule.avoid_genres}
    kw_terms = [k.casefold() for k in rule.keywords]
    text_terms = set(rule.overview_terms)

    pool = movies[movies["vote_count"] >= min_votes]
    if year_range is not None:
        lo, hi = sorted(year_range)
        years = pd.to_numeric(pool["release_year"], errors="coerce")
        pool = pool[(years >= lo) & (years <= hi)]
    if pool.empty:
        return pool.assign(mood_score=pd.Series(dtype=float), mood_reason=pd.Series(dtype=str)).head(0)

    scores, reasons = [], []
    for genres, keywords, overview in zip(pool["genres"], pool["keywords"], pool["overview"]):
        g_hits = [g for g in genres if g.casefold() in weights]
        g_score = sum(weights[g.casefold()] for g in g_hits)
        k_hits = [k for k in keywords if any(t in k.casefold() for t in kw_terms)]
        t_hits = sorted(text_terms & set(normalize_words(overview)))
        qualifies = bool(g_hits or k_hits)
        penalty = AVOID_PENALTY if any(g.casefold() in avoid for g in genres) else 0.0
        score = g_score + KEYWORD_POINTS * min(len(k_hits), KEYWORD_CAP) + TEXT_POINTS * min(len(t_hits), TEXT_CAP) - penalty
        scores.append(score if qualifies else -np.inf)
        bits = []
        if g_hits:
            bits.append("genres: " + ", ".join(g_hits))
        if k_hits:
            bits.append("keywords: " + ", ".join(k_hits[:3]))
        if t_hits:
            bits.append("description mentions: " + ", ".join(t_hits[:3]))
        reasons.append("; ".join(bits))

    out = pool.assign(mood_score=scores, mood_reason=reasons)
    out = out[out["mood_score"] > 0].copy()
    out["_wr"] = weighted_rating(out["vote_count"], out["vote_average"])
    out = out.sort_values(["mood_score", "_wr", "movie_id"], ascending=[False, False, True], kind="stable")
    return out.drop(columns="_wr").head(limit).reset_index(drop=True)
