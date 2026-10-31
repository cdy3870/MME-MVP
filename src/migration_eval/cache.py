"""Result cache (EVL-9): reuse a model's answer for the same request, model and settings.

The key is a hash of the provider, model and the full v0 request (messages, parameters, tools,
tool choice), so any change to the request or its settings misses the cache. A cached `complete`
wraps the real one, so nothing else in the harness knows the cache exists.

Caching and the noise floor pull in opposite directions: the noise floor (EVL-1) needs three
*independent* replays of the incumbent, and a cache would hand back the first answer three
times and report a floor of 1.0. So the key includes a `replica` number: the three incumbent
runs use replicas 0, 1 and 2, and a re-run of the same job reuses each of them.

Cached outputs derive from partner data. Where they are stored and for how long has to match
NFR-2 and NFR-3 (partner data is in use only inside the worker, and the retention rules).

Open decisions (see the notes on EVL-9):
    * where the cache lives (a directory inside the worker, deleted with it?) and its lifetime
    * whether failed results are cached (starting point: no, so a rate-limit error is retried)
    * the cache is only valid for the same catalog prices and harness code; changing the
      judge or the comparator does not invalidate model outputs, but changing `complete` does
    * the key does not name the logged provider, so use one cache directory per job (a request
      sent to a candidate is built differently when the traffic came from another provider)
    * a cached result keeps the latency it was first measured with, which is real but old

The directory and files are readable by the owner only, because the contents are partner data.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path

from migration_eval.models.record import ModelRequest
from migration_eval.replay import CompleteFn, ReplayResult


def cache_key(provider: str, model: str, request: ModelRequest, replica: int = 0) -> str:
    """A stable hex digest of the provider, model, request and replica number: the request
    dumped as JSON with sorted keys, then SHA-256.
    """
    payload = {
        "provider": provider,
        "model": model,
        "replica": replica,
        "request": request.model_dump(mode="json", by_alias=True, exclude_none=True),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResultCache:
    """A directory of cached results, one JSON file per key."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not key.isalnum():  # keys come from `cache_key`; never let one name another directory
            raise ValueError(f"bad cache key {key!r}")
        return self.directory / f"{key}.json"

    def get(self, key: str) -> ReplayResult | None:
        """The cached result, or None on a miss (or a file that cannot be read)."""
        path = self._path(key)  # a bad key is a bug, not a miss
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["unconverted_fields"] = tuple(data.get("unconverted_fields", ()))
            return ReplayResult(**data)
        except (OSError, ValueError, TypeError):  # missing, damaged, or from another version
            return None

    def put(self, key: str, result: ReplayResult) -> None:
        """Store a successful result. Write to a temp file and rename, so a crash never leaves
        half a file. Do not store a result with an `error`.
        """
        if result.error is not None:
            return
        path = self._path(key)
        fd, temp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")  # mode 0600
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dataclasses.asdict(result), handle)
            os.replace(temp, path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise


def cached(complete: CompleteFn, cache: ResultCache, replica: int = 0) -> CompleteFn:
    """Wrap `complete`: return the cached result when there is one, else call `complete` and
    store the answer. Pass `RateLimited` through untouched.
    """

    async def wrapper(provider: str, model: str, request: ModelRequest) -> ReplayResult:
        key = cache_key(provider, model, request, replica)
        hit = cache.get(key)
        if hit is not None:
            return hit
        result = await complete(provider, model, request)  # RateLimited propagates, nothing is stored
        cache.put(key, result)
        return result

    return wrapper
