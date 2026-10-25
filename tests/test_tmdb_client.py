import requests

from src.tmdb_client import TMDBClient, get_api_key, poster_url


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload or {}, headers or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# ---- missing credentials
def test_no_credentials_means_no_key():
    assert get_api_key(env={}, secrets=None) is None
    assert get_api_key(env={"TMDB_API_KEY": "   "}, secrets={}) is None


def test_secrets_take_precedence_over_environment():
    assert get_api_key(env={"TMDB_API_KEY": "from-env"}, secrets={"TMDB_API_KEY": "from-secrets"}) == "from-secrets"
    assert get_api_key(env={"TMDB_API_KEY": "from-env"}, secrets={}) == "from-env"


def test_broken_secrets_backend_is_ignored():
    class Broken:
        def get(self, key):
            raise FileNotFoundError("no secrets.toml")

    assert get_api_key(env={"TMDB_API_KEY": "k"}, secrets=Broken()) == "k"


def test_client_without_key_never_touches_network():
    class Boom:
        def get(self, *a, **k):
            raise AssertionError("network must not be used without a key")

    client = TMDBClient(None, session=Boom())
    assert not client.enabled and not client.has_key
    assert client.get_poster_path(1) is None
    assert client.get_poster_urls([1, 2]) == {1: None, 2: None}


# ---- API behaviour
def test_success_and_caching():
    session = FakeSession(FakeResponse(200, {"poster_path": "/abc.jpg"}))
    client = TMDBClient("a" * 32, session=session)
    assert client.get_poster_path(7) == "/abc.jpg"
    assert client.get_poster_path(7) == "/abc.jpg"
    assert len(session.calls) == 1  # second call served from cache
    assert session.calls[0][1]["params"]["api_key"] == "a" * 32
    assert client.get_poster_urls([7]) == {7: "https://image.tmdb.org/t/p/w342/abc.jpg"}


def test_v4_token_uses_bearer_header():
    session = FakeSession(FakeResponse(200, {"poster_path": "/x.jpg"}))
    TMDBClient("eyJ" + "x" * 50, session=session).get_poster_path(1)
    assert session.calls[0][1]["headers"]["Authorization"].startswith("Bearer eyJ")
    assert "params" not in session.calls[0][1]


def test_missing_poster_and_404_are_negative_cached():
    session = FakeSession(FakeResponse(404))
    client = TMDBClient("k", session=session)
    assert client.get_poster_path(1) is None and client.get_poster_path(1) is None
    assert len(session.calls) == 1


def test_null_poster_path():
    client = TMDBClient("k", session=FakeSession(FakeResponse(200, {"poster_path": None})))
    assert client.get_poster_path(1) is None


def test_timeout_and_connection_errors_fall_back_gracefully():
    for error in (requests.Timeout("slow"), requests.ConnectionError("down")):
        client = TMDBClient("k", session=FakeSession(error))
        assert client.get_poster_path(1) is None


def test_invalid_json_is_handled():
    class BadJson(FakeResponse):
        def json(self):
            raise ValueError("not json")

    assert TMDBClient("k", session=FakeSession(BadJson(200))).get_poster_path(1) is None


def test_unauthorized_disables_client():
    session = FakeSession(FakeResponse(401))
    client = TMDBClient("bad", session=session)
    assert client.get_poster_path(1) is None
    assert not client.enabled and "401" in client.disabled_reason
    client.get_poster_path(2)
    assert len(session.calls) == 1  # no further requests with a rejected key


def test_rate_limit_pauses_requests_until_retry_after():
    clock = Clock()
    session = FakeSession(FakeResponse(429, headers={"Retry-After": "30"}), FakeResponse(200, {"poster_path": "/ok.jpg"}))
    client = TMDBClient("k", session=session, clock=clock)
    assert client.get_poster_path(1) is None
    assert client.get_poster_path(2) is None and len(session.calls) == 1  # paused, no request made
    clock.now += 31
    assert client.get_poster_path(2) == "/ok.jpg" and len(session.calls) == 2


def test_cache_expires():
    clock = Clock()
    session = FakeSession(FakeResponse(200, {"poster_path": "/a.jpg"}))
    client = TMDBClient("k", session=session, ttl=10, clock=clock)
    client.get_poster_path(1)
    clock.now += 11
    client.get_poster_path(1)
    assert len(session.calls) == 2


def test_poster_url_helper():
    assert poster_url(None) is None and poster_url("") is None
    assert poster_url("/p.jpg", "w500") == "https://image.tmdb.org/t/p/w500/p.jpg"
