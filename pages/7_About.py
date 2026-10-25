import streamlit as st

from src import tmdb_client, ui

engine = ui.get_engine()

st.title("About CineMatch AI")
st.markdown(f"""
**CineMatch AI** is an academic / portfolio project showing an end-to-end content-based recommender: raw CSVs →
cleaning → feature engineering → TF-IDF → cosine similarity → explainable recommendations in a Streamlit app.

### Objectives
- Recommend movies by content similarity and explain *why* each one is suggested.
- Demonstrate a real ML pipeline with honest evaluation, caching and automated tests.
- Offer a smooth discovery experience: search, filters, moods, details, watchlist.

### Technology
Python · Streamlit · pandas / NumPy · scikit-learn (`TfidfVectorizer`, `cosine_similarity`) · Plotly · SQLite ·
Requests (optional TMDB posters) · pytest.

### Method in brief
1. Merge `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` on the movie id; parse the JSON columns safely.
2. Build one text per movie from genres, keywords, director, top cast and overview (documented repetition weights).
3. Fit `TfidfVectorizer` (unigrams + bigrams, English stop words, ≤20,000 features) → sparse matrix.
4. For a chosen movie, compute cosine similarity to all others, filter, rank deterministically, and explain with verified shared attributes.

### Data attribution
Movie data: the **TMDB 5000 Movie Dataset** ([Kaggle]({ui.KAGGLE_URL})), derived from [The Movie Database (TMDB)](https://www.themoviedb.org/).
Please review the dataset page for its current terms. No dataset files are distributed with this project.

*{tmdb_client.ATTRIBUTION}* Posters are only requested from TMDB when you configure an API key; otherwise a placeholder is shown.

### Poster mode right now
**{ui.poster_mode()}** — the recommendation engine never needs the API.

### Limitations
Similarity scores describe content overlap, not personal enjoyment. The data is a static snapshot, there is no
personalisation, and the watchlist is a local single-user store. See **ML Insights → Limitations** for details.

### Setup summary
`pip install -r requirements.txt` → put the two CSVs in `data/` → `streamlit run app.py`. Full instructions are in the README.
""")
st.caption(f"Loaded {len(engine.movies):,} movies from `{engine.data_dir}`.")
