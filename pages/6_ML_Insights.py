import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import evaluation, ui
from src.preprocessing import FeatureWeights

engine = ui.get_engine()
rec, movies, report = engine.recommender, engine.movies, engine.report
n_movies = len(movies)

st.title("ML Insights & Model Information")
st.caption("Everything on this page is computed from the loaded dataset and the fitted model.")
tab_concepts, tab_data, tab_model, tab_eval, tab_limits = st.tabs(
    ["Concepts", "Dataset", "Model", "Evaluation", "Limitations"])

# --------------------------------------------------------------------------- concepts
with tab_concepts:
    st.markdown("""
**Content-based filtering** recommends items that are *similar in content* to one you already like. It uses no other
users' behaviour: every movie is described by its own metadata (genres, keywords, plot overview, director, cast).

**Why TF-IDF?** Each movie becomes a text document. TF-IDF turns it into a vector with one number per vocabulary term:
a term scores high when it appears often in *this* movie (term frequency) but in few movies overall (inverse document
frequency). Common words carry little weight; distinctive ones such as `gotham` or `christopher_nolan` carry a lot.""")
    st.latex(r"\mathrm{tfidf}(t,d)=\mathrm{tf}(t,d)\times\left(\ln\frac{1+N}{1+\mathrm{df}(t)}+1\right)")
    st.markdown("**Why cosine similarity?** It compares the *direction* of two vectors, so long and short descriptions "
                "can still be compared fairly. With L2-normalised vectors it is just their dot product.")
    st.latex(r"\cos(\theta)=\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert}")
    st.markdown("**Which features contribute?** The combined text repeats each field's tokens by a documented weight, so "
                "genres and keywords influence the vector more than the free-text overview:")
    st.dataframe(pd.DataFrame([FeatureWeights().as_dict()], index=["repetitions"]).T.rename_axis("field"), width=320)
    st.caption("Genres/director/cast are stored as single tokens (e.g. `science_fiction`, `sam_worthington`) so names "
               "and multi-word genres stay intact; keyword phrases are split into words.")

# ------------------------------------------------------------------------------ data
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in movies file", f"{report.movies_raw_rows:,}")
    c2.metric("Rows in credits file", f"{report.credits_raw_rows:,}")
    c3.metric("Unique movies processed", f"{n_movies:,}")
    c4.metric("Columns (movies / credits)", f"{len(report.raw_missing_movies)} / {len(report.raw_missing_credits)}")

    st.markdown("**Cleaning actions taken**")
    st.dataframe(pd.DataFrame({
        "Check": ["Rows with unusable id", "Duplicate ids removed", "Same title + release date removed",
                  "Movies without a credits row", "Unparseable release dates", "Malformed JSON cells (all columns)",
                  "Different movies that share a title"],
        "Count": [report.rows_dropped_bad_id, report.duplicate_ids_removed, report.duplicate_title_date_removed,
                  report.rows_without_credits, report.invalid_release_dates, sum(report.malformed_json.values()),
                  report.titles_shared_by_different_movies]}), hide_index=True, width=520)

    miss = pd.DataFrame({
        "Field": ["Overview", "Release date", "Runtime", "Genres", "Keywords", "Director", "Rating (0 votes)"],
        "Missing / empty": [report.missing_overview, report.missing_release_date, report.missing_runtime,
                            report.empty_genres, report.empty_keywords, report.missing_director, report.unrated_movies]})
    miss["Share of movies"] = (miss["Missing / empty"] / max(n_movies, 1)).round(4)
    fig = px.bar(miss, x="Field", y="Missing / empty", hover_data=["Share of movies"], title="Missing or empty values after cleaning",
                 color_discrete_sequence=["#8b5cf6"])
    ui.show_chart(fig)
    raw = pd.DataFrame({"Column": list(report.raw_missing_movies), "Null cells": list(report.raw_missing_movies.values())})
    with st.expander("Null cells per column in the raw movies file"):
        st.dataframe(raw[raw["Null cells"] > 0], hide_index=True)

    g_counts = movies.explode("genres")["genres"].value_counts().rename_axis("Genre").reset_index(name="Movies")
    ui.show_chart(px.bar(g_counts.sort_values("Movies"), x="Movies", y="Genre", orientation="h",
                         title="Movies per genre (a movie can have several)", color_discrete_sequence=["#22d3ee"]))
    decades = (movies["release_year"].dropna().astype(int) // 10 * 10).value_counts().sort_index()
    ui.show_chart(px.bar(x=decades.index.astype(str) + "s", y=decades.values, labels={"x": "Decade", "y": "Movies"},
                         title="Movies per decade", color_discrete_sequence=["#8b5cf6"]).update_layout(height=340))
    rated = movies.dropna(subset=["vote_average"])
    rated = rated[rated["vote_count"] >= 10]
    ui.show_chart(px.histogram(rated, x="vote_average", nbins=40, title=f"Rating distribution ({len(rated):,} movies with ≥10 votes)",
                               labels={"vote_average": "TMDB average rating"}, color_discrete_sequence=["#22d3ee"]).update_layout(height=340))

# ----------------------------------------------------------------------------- model
with tab_model:
    m = rec.matrix
    c1, c2, c3 = st.columns(3)
    c1.metric("TF-IDF matrix", f"{m.shape[0]:,} × {m.shape[1]:,}")
    c2.metric("Non-zero entries", f"{m.nnz:,}")
    c3.metric("Density", f"{m.nnz / (m.shape[0] * m.shape[1]):.3%}")
    st.caption("Stored as a sparse matrix. Similarities are computed one movie at a time, so the full N×N matrix never sits in memory.")
    with st.expander("TfidfVectorizer parameters"):
        st.code(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in rec.params.items()}, indent=2), language="json")

    mean_w = np.asarray(m.mean(axis=0)).ravel()
    top = np.argsort(-mean_w, kind="stable")[:20]
    terms = rec.vectorizer.get_feature_names_out()[top]
    ui.show_chart(px.bar(x=mean_w[top][::-1], y=terms[::-1], orientation="h", labels={"x": "Mean TF-IDF weight", "y": "Term"},
                         title="Terms with the highest average TF-IDF weight", color_discrete_sequence=["#8b5cf6"]).update_layout(height=480))

    st.markdown("**Examples of feature matches** (the three most-voted movies in the dataset, top 3 matches each)")
    seeds = movies.sort_values(["vote_count", "movie_id"], ascending=[False, True]).head(3)
    rows = []
    for _, s in seeds.iterrows():
        for _, r in rec.recommend(movie_id=int(s["movie_id"]), top_n=3).iterrows():
            rows.append({"Movie": s["label"], "Recommended": r["label"], "Similarity": round(r["similarity_score"], 3),
                         "Why (verified shared features)": r["explanation"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# -------------------------------------------------------------------------- evaluation
@st.cache_data(show_spinner="Evaluating franchise retrieval…")
def run_proxy(_rec, n: int):
    return evaluation.evaluate_franchise_proxy(_rec)


@st.cache_data(show_spinner="Computing coverage and diversity for every movie…")
def run_coverage(_rec, n: int):
    return evaluation.coverage_and_diversity(_rec, k=10)


with tab_eval:
    st.markdown("""
**What can and cannot be measured.** The TMDB 5000 files contain no user behaviour, so there is no ground truth for
*“would this person like it?”*. Rather than invent one, this page reports three limited, clearly labelled checks.
Similarity scores are **content-overlap scores, not probabilities of enjoyment**.""")
    st.markdown("1. **Franchise-retrieval proxy (Precision@K / Recall@K).** Relevant = other movies with the same title stem "
                "(e.g. *Toy Story*, *Toy Story 2*). Titles are not a model input, so the label is independent of the features, "
                "but it only tests sequel/series retrieval and contains some noise.\n"
                "2. **Coverage & diversity** (no labels needed).\n"
                "3. **Manual relevance check** — see the guide at the bottom of this tab.")
    if st.button("Run evaluation on the full catalogue", type="primary", icon=":material/play_arrow:"):
        st.session_state["run_eval"] = True
    if not st.session_state.get("run_eval"):
        st.caption("Takes roughly 10–20 seconds the first time; results are cached afterwards.")
    else:
        summary, detail, info = run_proxy(rec, n_movies)
        if summary.empty:
            st.warning("No franchise groups were found in this dataset, so the proxy cannot be computed.")
        else:
            st.markdown(f"**Franchise proxy:** {info['queries']} query movies in {info['groups']} groups.")
            st.dataframe(summary.round(3), hide_index=True, width=520)
            st.caption("Precision@K is capped by group size: a movie with only 1–2 siblings can never reach 100% at K=10. "
                       "Recall@K is the more informative number here.")
            strong, weak = evaluation.strong_and_weak_examples(detail)
            s1, s2 = st.columns(2)
            s1.markdown("**Strong examples** — a sibling is ranked first")
            s1.dataframe(strong[["title", "n_relevant", "P@5", "top1_score"]].round(3), hide_index=True)
            s2.markdown("**Weak examples** — no sibling in the top 20")
            s2.dataframe(weak[["title", "n_relevant", "top1_score"]].round(3), hide_index=True)
            st.caption("Some “weak” cases are label noise rather than model errors (e.g. *Halloween III* shares a title stem "
                       "with the Michael Myers films but is a different story).")

        cov = run_coverage(rec, n_movies)
        st.markdown(f"**Coverage & diversity (top-{cov['k']} lists for all {n_movies:,} movies)**")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Catalogue coverage", f"{cov['coverage']:.1%}", help="Share of movies that appear in at least one top-K list.")
        d2.metric("Never recommended", f"{cov['never_recommended']:,}")
        d3.metric("Intra-list diversity", f"{cov['mean_intra_list_diversity']:.2f}",
                  help=f"Mean 1 − cosine between items in the same list (sample of {cov['diversity_sample_size']}). Higher = more varied.")
        d4.metric("Median top-K similarity", f"{cov['median_topk_score']:.2f}")
        st.caption(f"Most-recommended movie appears in {cov['max_times_recommended']} lists. Median vote count: catalogue "
                   f"{cov['median_votes_catalogue']:.0f} vs recommended {cov['median_votes_recommended']:.0f}, so there is "
                   "no strong tilt toward heavily-voted titles.")
        ui.show_chart(px.histogram(x=cov["top_scores"][:, 0], nbins=40, title="Similarity of each movie's best match",
                                   labels={"x": "Top-1 content similarity"}, color_discrete_sequence=["#22d3ee"]).update_layout(height=320))
    with st.expander("How to do a manual relevance check"):
        st.markdown("Open **Find Similar Movies** for a film you know well. For each of the top 5 results ask: *would I "
                    "describe this as the same kind of movie?* Count the yes answers (manual Precision@5) and read the "
                    "“Why” line to check the explanation is truthful. Repeat for ~10 movies across genres, including "
                    "obscure ones with short descriptions, and note strong and weak cases.")

# ------------------------------------------------------------------------- limitations
with tab_limits:
    st.markdown(f"""
**Known limitations**
- No personalisation: results depend only on the selected movie's content, never on your taste history.
- The dataset is a fixed snapshot (about {n_movies:,} movies); newer films are absent and popularity/ratings are not live.
- {report.empty_keywords} movies have no keywords and {report.missing_overview} have no overview, so their matches are weaker.
- TF-IDF matches *words*, not meaning: synonyms and tone are not understood, and generic overviews produce generic matches.
- Only the top billed cast and one director are used; other crew, studios and franchises are not modelled.
- Mood matching is rule-based and approximate.
- The evaluation proxy measures sequel retrieval only; real satisfaction would need user feedback data.

**Possible improvements**
- Sentence embeddings for overviews, blended with TF-IDF.
- Collaborative filtering or hybrid models once user ratings are collected.
- Diversity-aware re-ranking (e.g. maximal marginal relevance).
- A collected set of human-labelled relevance judgements for proper Precision@K.
""")
