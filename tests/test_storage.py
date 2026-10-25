import pytest

from src.storage import LocalStore


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "sub" / "cm.db")  # parent dir is created on demand


def test_add_remove_contains(store):
    assert not store.contains(1, "watchlist")
    store.add(1, "watchlist")
    store.add(1, "watchlist")  # idempotent
    assert store.contains(1, "watchlist") and store.list_ids("watchlist") == [1]
    store.remove(1, "watchlist")
    assert not store.contains(1, "watchlist")
    store.remove(1, "watchlist")  # removing twice is harmless


def test_lists_are_independent_and_newest_first(store):
    for i in (1, 2, 3):
        store.add(i, "watchlist")
    store.add(2, "favorite")
    assert store.list_ids("watchlist") == [3, 2, 1]
    assert store.list_ids("favorite") == [2]


def test_toggle(store):
    assert store.toggle(5, "favorite") is True
    assert store.toggle(5, "favorite") is False
    assert store.list_ids("favorite") == []


def test_data_persists_across_instances(tmp_path):
    path = tmp_path / "cm.db"
    first = LocalStore(path)
    first.add(42, "watchlist")
    first.set_rating(42, 4)
    second = LocalStore(path)
    assert second.list_ids("watchlist") == [42] and second.get_rating(42) == 4


def test_ratings_crud(store):
    assert store.get_rating(9) is None
    store.set_rating(9, 3)
    store.set_rating(9, 5)  # overwrite
    assert store.get_rating(9) == 5 and store.all_ratings() == {9: 5}
    store.clear_rating(9)
    assert store.get_rating(9) is None


@pytest.mark.parametrize("bad", [0, 6, -1, 3.5, "4", True, None])
def test_invalid_ratings_rejected(store, bad):
    with pytest.raises(ValueError):
        store.set_rating(1, bad)


def test_invalid_kind_rejected(store):
    for call in (store.add, store.remove, store.contains, store.toggle):
        with pytest.raises(ValueError):
            call(1, "bogus")
    with pytest.raises(ValueError):
        store.list_ids("bogus")


def test_env_var_selects_database(monkeypatch, tmp_path):
    monkeypatch.setenv("CINEMATCH_DB_PATH", str(tmp_path / "env.db"))
    LocalStore().add(1, "watchlist")
    assert (tmp_path / "env.db").exists()
