"""Optional TMDB API client used ONLY for poster images.

* Without an API key this module never touches the network.
* Keys come from the environment / Streamlit secrets, never from source code.
* Responses are cached in memory (positive and negative), calls have a timeout, a
  429 response pauses all requests until ``Retry-After`` has passed, and a 401 disables
  the client so a bad key is not retried on every page view.

Attribution required by TMDB: "This product uses the TMDB API but is not endorsed or
certified by TMDB."
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Mapping

import requests

API_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"
KEY_ENV = "TMDB_API_KEY"
ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."


def get_api_key(env: Mapping[str, str] | None = None, secrets: Mapping[str, object] | None = None) -> str | None:
    """Streamlit secrets take precedence over environment variables. Blank -> None."""
    env = os.environ if env is None else env
    for source in (secrets, env):
        if source is None:
            continue
        try:
            value = source.get(KEY_ENV)
        except Exception:  # secrets backends can raise when no secrets file exists
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def poster_url(path: str | None, size: str = "w342") -> str | None:
    if not path or not isinstance(path, str):
        return None
    return f"{IMAGE_BASE}/{size}/{path.lstrip('/')}"


class TMDBClient:
    def __init__(
        self,
        api_key: str | None,
        *,
        session: requests.Session | None = None,
        timeout: float = 5.0,
        ttl: float = 86400.0,
        negative_ttl: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._key = (api_key or "").strip() or None
        self._session = session or requests.Session()
        self._timeout, self._ttl, self._neg_ttl, self._clock = timeout, ttl, negative_ttl, clock
        self._cache: dict[int, tuple[float, str | None]] = {}
        self._lock = threading.Lock()
        self._blocked_until = 0.0
        self.disabled_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self._key is not None and self.disabled_reason is None

    @property
    def has_key(self) -> bool:
        return self._key is not None

    def _request_args(self) -> dict:
        # v4 "read access tokens" are long JWTs (start with 'eyJ'); v3 keys are 32 hex chars.
        if self._key and self._key.startswith("eyJ"):
            return {"headers": {"Authorization": f"Bearer {self._key}", "accept": "application/json"}}
        return {"params": {"api_key": self._key}, "headers": {"accept": "application/json"}}

    def get_poster_path(self, movie_id: int) -> str | None:
        """Poster path (e.g. '/abc.jpg') for a TMDB movie id, or None."""
        if not self.enabled:
            return None
        now = self._clock()
        with self._lock:
            hit = self._cache.get(int(movie_id))
            if hit and hit[0] > now:
                return hit[1]
            if now < self._blocked_until:
                return None
        path, ttl = None, self._neg_ttl
        try:
            resp = self._session.get(f"{API_BASE}/movie/{int(movie_id)}", timeout=self._timeout, **self._request_args())
            if resp.status_code == 200:
                value = resp.json().get("poster_path")
                if isinstance(value, str) and value:
                    path, ttl = value, self._ttl
            elif resp.status_code == 401:
                self.disabled_reason = "TMDB rejected the API key (HTTP 401)."
            elif resp.status_code == 429:
                try:
                    wait = float(resp.headers.get("Retry-After", 10))
                except (TypeError, ValueError):
                    wait = 10.0
                with self._lock:
                    self._blocked_until = self._clock() + max(wait, 1.0)
                return None  # don't negative-cache: the movie itself is fine
        except (requests.RequestException, ValueError):
            pass  # timeouts, connection errors, bad JSON -> graceful fallback
        with self._lock:
            self._cache[int(movie_id)] = (self._clock() + ttl, path)
        return path

    def get_poster_urls(self, movie_ids: Iterable[int], size: str = "w342", workers: int = 8) -> dict[int, str | None]:
        ids = list(dict.fromkeys(int(i) for i in movie_ids))
        if not self.enabled or not ids:
            return {i: None for i in ids}
        with ThreadPoolExecutor(max_workers=min(workers, len(ids))) as pool:
            paths = list(pool.map(self.get_poster_path, ids))
        return {i: poster_url(p, size) for i, p in zip(ids, paths)}
