"""Honest, reproducible evaluation helpers for the recommender.

The TMDB 5000 files contain NO user-behaviour data (no ratings per user, no watch
history), so a true "did the user like it" ground truth does not exist here. We therefore
report only metrics that can be defended, each with its limitations spelled out:

1. Franchise-retrieval proxy (Precision@K / Recall@K)
   Relevant items for a movie = other movies whose *title stem* is identical
   ("Toy Story", "Toy Story 2", "Toy Story 3" -> stem "toy story"). The stem is derived
   from the title only, and titles are NOT a model input, so the label is independent of
   the features. It measures one narrow thing: can the model find sequels / series.
2. Catalogue coverage and intra-list diversity (no labels needed).
3. Distribution of top-K similarity scores (how confident the matches are).

None of these measure whether a person would enjoy a recommendation.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .recommender import MovieRecommender

_ROMAN = r"(?:ii|iii|iv|v|vi|vii|viii|ix|x)"
_TRAILING = re.compile(rf"(?:\s+(?:part|chapter|volume|vol\.?)\s+\w+|\s+\d+|\s+{_ROMAN})+$")
_MIN_STEM_CHARS = 5


def franchise_stem(title: str) -> str:
    """'Toy Story 3' -> 'toy story'; 'Harry Potter and the Order of ...' is NOT merged (documented)."""
    if not isinstance(title, str):
        return ""
    t = title.casefold().strip()
    t = re.split(r"\s*[:\u2013\u2014]\s*|\s+-\s+", t)[0]  # drop subtitle
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _TRAILING.sub("", t).strip()
    t = re.sub(r"^(?:the|a|an)\s+", "", t)
    return t if len(t) >= _MIN_STEM_CHARS else ""


def build_franchise_groups(movies: pd.DataFrame, min_size: int = 2) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for movie_id, title in zip(movies["movie_id"], movies["title"]):
        stem = franchise_stem(title)
        if stem:
            groups[stem].append(int(movie_id))
    return {s: ids for s, ids in groups.items() if len(ids) >= min_size}


def precision_recall_at_k(recommended: list[int], relevant: set[int], k: int) -> tuple[float, float]:
    """Standard definitions. Returns (precision@k, recall@k); recall is 0.0 if nothing is relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = recommended[:k]
    hits = sum(1 for m in top if m in relevant)
    return hits / k, (hits / len(relevant) if relevant else 0.0)


def evaluate_franchise_proxy(rec: MovieRecommender, ks: tuple[int, ...] = (5, 10, 20)) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (summary_by_k, per_query_details, info)."""
    groups = build_franchise_groups(rec.movies)
    kmax = max(ks)
    rows = []
    for stem, ids in groups.items():
        for qid in ids:
            relevant = set(ids) - {qid}
            recs = rec.recommend(movie_id=qid, top_n=kmax, explain=False)
            rec_ids = recs["movie_id"].tolist()
            row = {"stem": stem, "movie_id": qid, "title": rec.movies["label"].iat[rec.index_for_id(qid)],
                   "n_relevant": len(relevant),
                   "top1_score": float(recs["similarity_score"].iat[0]) if len(recs) else 0.0}
            for k in ks:
                p, r = precision_recall_at_k(rec_ids, relevant, k)
                row[f"P@{k}"], row[f"R@{k}"] = p, r
                row[f"hit@{k}"] = float(p > 0)
            first = next((i + 1 for i, m in enumerate(rec_ids) if m in relevant), None)
            row["first_hit_rank"] = first
            rows.append(row)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), detail, {"groups": 0, "queries": 0}
    summary = pd.DataFrame({
        "K": list(ks),
        "Precision@K": [detail[f"P@{k}"].mean() for k in ks],
        "Recall@K": [detail[f"R@{k}"].mean() for k in ks],
        "Hit rate@K": [detail[f"hit@{k}"].mean() for k in ks],
    })
    info = {"groups": len(groups), "queries": len(detail), "largest_group": max(len(v) for v in groups.values())}
    return summary, detail, info


def coverage_and_diversity(rec: MovieRecommender, k: int = 10, chunk: int = 500,
                           diversity_sample: int = 500, seed: int = 42) -> dict:
    """Catalogue coverage, intra-list diversity and top-K similarity scores for ALL movies as queries.

    Coverage = share of movies appearing in at least one other movie's top-K list.
    Intra-list diversity = mean (1 - cosine) between items inside a list (higher = more varied).
    """
    n = len(rec)
    k = min(k, n - 1)
    if k < 1:
        raise ValueError("Need at least two movies to evaluate.")
    mat = rec.matrix
    top_idx = np.zeros((n, k), dtype=np.int64)
    top_scores = np.zeros((n, k), dtype=float)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = (mat[start:stop] @ mat.T).toarray()
        sims[np.arange(stop - start), np.arange(start, stop)] = -1.0  # exclude self
        order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        top_idx[start:stop] = order
        top_scores[start:stop] = np.take_along_axis(sims, order, axis=1)

    recommended = np.unique(top_idx)
    counts = np.bincount(top_idx.ravel(), minlength=n)

    rng = np.random.default_rng(seed)
    sample = rng.choice(n, size=min(diversity_sample, n), replace=False)
    ild = []
    for q in sample:
        block = mat[top_idx[q]]
        s = (block @ block.T).toarray()
        iu = np.triu_indices(k, 1)
        if iu[0].size:
            ild.append(1.0 - float(s[iu].mean()))

    votes = rec.movies["vote_count"].to_numpy()
    return {
        "k": k,
        "coverage": float(recommended.size / n),
        "never_recommended": int(n - recommended.size),
        "max_times_recommended": int(counts.max()),
        "mean_intra_list_diversity": float(np.mean(ild)) if ild else float("nan"),
        "diversity_sample_size": int(len(sample)),
        "mean_top1_score": float(top_scores[:, 0].mean()),
        "median_topk_score": float(np.median(top_scores)),
        "top_scores": top_scores,
        "median_votes_catalogue": float(np.median(votes)),
        "median_votes_recommended": float(np.median(votes[recommended])) if recommended.size else float("nan"),
    }


def strong_and_weak_examples(detail: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strong = franchise members found at rank 1 with the highest precision; weak = no hit in top 20."""
    if detail.empty:
        return detail, detail
    strong = (detail[detail["first_hit_rank"] == 1]
              .sort_values(["P@5", "top1_score", "movie_id"], ascending=[False, False, True]).head(n))
    weak = (detail[detail["hit@20"] == 0]
            .sort_values(["n_relevant", "movie_id"], ascending=[False, True]).head(n))
    return strong.reset_index(drop=True), weak.reset_index(drop=True)


def render_markdown_report(rec: MovieRecommender, k: int = 10) -> str:
    summary, detail, info = evaluate_franchise_proxy(rec)
    cov = coverage_and_diversity(rec, k=k)
    strong, weak = strong_and_weak_examples(detail)
    lines = ["# CineMatch AI - evaluation report", "",
             f"Catalogue size: {len(rec)} movies; TF-IDF matrix {rec.matrix.shape[0]} x {rec.matrix.shape[1]}.", "",
             "## Franchise-retrieval proxy",
             f"{info.get('queries', 0)} query movies in {info.get('groups', 0)} title-stem groups. "
             "Relevant = other movies with the same title stem. This is a narrow proxy (sequel retrieval), "
             "not a measure of user satisfaction.", "",
             summary.round(3).to_markdown(index=False) if not summary.empty else "_no groups found_", "",
             "## Coverage and diversity (all movies as queries)",
             f"- Top-{cov['k']} catalogue coverage: {cov['coverage']:.1%} ({cov['never_recommended']} movies never recommended)",
             f"- Most-recommended movie appears in {cov['max_times_recommended']} lists",
             f"- Mean intra-list diversity (sample of {cov['diversity_sample_size']}): {cov['mean_intra_list_diversity']:.3f}",
             f"- Mean top-1 similarity: {cov['mean_top1_score']:.3f}; median top-{cov['k']} similarity: {cov['median_topk_score']:.3f}",
             f"- Median vote count: catalogue {cov['median_votes_catalogue']:.0f} vs recommended {cov['median_votes_recommended']:.0f}", "",
             "## Strong examples (franchise sibling at rank 1)",
             strong[["title", "n_relevant", "P@5", "top1_score"]].round(3).to_markdown(index=False) if len(strong) else "_none_", "",
             "## Weak examples (no franchise sibling in top 20)",
             weak[["title", "n_relevant", "top1_score"]].round(3).to_markdown(index=False) if len(weak) else "_none_", ""]
    return "\n".join(lines)


def main() -> None:  # python -m src.evaluation --out docs/evaluation_report.md
    from .data_loader import load_raw_data
    from .preprocessing import build_movie_table

    parser = argparse.ArgumentParser(description="Evaluate the CineMatch recommender.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=None, help="write the markdown report to this path")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    movies_raw, credits_raw = load_raw_data(args.data_dir)
    movies, _ = build_movie_table(movies_raw, credits_raw)
    report = render_markdown_report(MovieRecommender(movies), k=args.k)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
