import streamlit as st

from src import ui

engine = ui.get_engine()
store = ui.get_store()

st.title("Watchlist & Favorites")
st.info("Your lists are saved in a local SQLite file on the machine running this app. This single-user setup is meant "
        "for demonstration. If you deploy publicly, all visitors would share one file, so add user accounts and "
        "per-user storage before collecting real preferences.", icon=":material/lock:")

lists = {kind: store.list_ids(kind) for kind in ("watchlist", "favorite")}
ratings = store.all_ratings()
tab_w, tab_f, tab_r = st.tabs([f"Watchlist ({len(lists['watchlist'])})", f"Favorites ({len(lists['favorite'])})",
                               f"My ratings ({len(ratings)})"])


def show(ids: list[int], key: str, remove_from: str, empty_msg: str) -> None:
    if not ids:
        st.info(empty_msg)
        return
    frame = engine.frame_for(ids)
    if len(frame) < len(ids):
        st.warning(f"{len(ids) - len(frame)} saved movie(s) are not in the currently loaded dataset and are hidden.")
    ui.render_grid(frame, key=key, n_cols=5, remove_from=remove_from)


with tab_w:
    show(lists["watchlist"], "wl", "watchlist", "Your watchlist is empty. Open a movie's **Details** page and choose “Add to watchlist”.")
with tab_f:
    show(lists["favorite"], "fav", "favorite", "No favorites yet. Open a movie's **Details** page and choose “Mark as favorite”.")
with tab_r:
    show(list(ratings), "rt", "rating", "You have not rated any movies yet. Rate them with the stars on a movie's Details page.")
