"""Minimal fal.ai queue client used by the drama pipeline.

Only the documented queue REST API is used:

* ``POST https://queue.fal.run/{endpoint}`` submits a request and returns
  ``request_id``, ``status_url``, ``response_url`` and ``cancel_url``.
* ``GET {status_url}`` reports ``IN_QUEUE``, ``IN_PROGRESS`` or ``COMPLETED``
  (a failed request is ``COMPLETED`` with ``error``/``error_type``).
* ``GET {response_url}`` returns the model output.

Every call here spends money on the user's fal.ai account, so nothing in this
module retries a submission that may already have been accepted.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

import requests
from loguru import logger

from app.config import config

QUEUE_BASE_URL = "https://queue.fal.run"
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 30 * 60
HTTP_TIMEOUT_SECONDS = 60
# Retry only idempotent reads (status/result/download), never the paid submit.
_READ_RETRIES = 4
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class FalError(RuntimeError):
    """A fal.ai request failed."""


class FalUnconfirmedSubmitError(FalError):
    """The submit may have reached fal.ai; resubmitting could charge twice."""


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """Read ``app.fal_api_key`` from config.toml, then the ``FAL_KEY`` env var."""
    settings = config.app if settings is None else settings
    configured = str(settings.get("fal_api_key", "") or "").strip()
    return configured or os.getenv("FAL_KEY", "").strip()


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _redact(text: Any, api_key: str) -> str:
    message = str(text)
    if api_key:
        message = message.replace(api_key, "***")
    return message[:800]


def _error_detail(response: requests.Response, api_key: str) -> str:
    try:
        body = response.json()
    except ValueError:
        return _redact(response.text, api_key)
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body.get("message")
        if detail:
            return _redact(detail, api_key)
    return _redact(body, api_key)


def file_to_data_uri(path: str) -> str:
    """Encode a local image as a data URI, which fal accepts for file inputs."""
    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


class FalClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: requests.Session | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.api_key = (api_key if api_key is not None else get_api_key()).strip()
        if not self.api_key:
            raise FalError(
                "fal.ai API key is missing: set app.fal_api_key in config.toml "
                "or the FAL_KEY environment variable"
            )
        self.session = session or requests.Session()
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self._sleep = sleep
        self._clock = clock

    # -- low level -------------------------------------------------------
    def _get(self, url: str, **kwargs) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(_READ_RETRIES + 1):
            try:
                response = self.session.get(
                    url, headers=_headers(self.api_key), timeout=HTTP_TIMEOUT_SECONDS, **kwargs
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code not in _RETRYABLE_STATUS:
                    return response
                last_error = FalError(
                    f"HTTP {response.status_code}: {_error_detail(response, self.api_key)}"
                )
            if attempt < _READ_RETRIES:
                self._sleep(min(2**attempt, 20))
        raise FalError(f"fal.ai read failed for {url}: {_redact(last_error, self.api_key)}")

    def submit(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Submit one paid request. Never retried automatically."""
        endpoint = endpoint.strip().strip("/")
        if not endpoint:
            raise FalError("fal.ai endpoint id must not be empty")
        url = f"{QUEUE_BASE_URL}/{endpoint}"
        try:
            response = self.session.post(
                url,
                json=dict(payload),
                headers={**_headers(self.api_key), "X-Fal-No-Retry": "1"},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise FalUnconfirmedSubmitError(
                f"fal.ai submit to {endpoint} did not get a response; it may still "
                "run and be billed. Check https://fal.ai/dashboard/requests before "
                f"retrying. Detail: {_redact(exc, self.api_key)}"
            ) from exc
        if response.status_code >= 500:
            raise FalUnconfirmedSubmitError(
                f"fal.ai submit to {endpoint} returned HTTP {response.status_code}; "
                "check the fal.ai dashboard before retrying: "
                f"{_error_detail(response, self.api_key)}"
            )
        if response.status_code >= 400:
            raise FalError(
                f"fal.ai rejected the request to {endpoint} (HTTP {response.status_code}): "
                f"{_error_detail(response, self.api_key)}"
            )
        body = response.json()
        for field in ("request_id", "status_url", "response_url"):
            if not body.get(field):
                raise FalError(f"fal.ai submit response is missing {field}: {body}")
        logger.info(f"fal.ai request queued: endpoint={endpoint}, id={body['request_id']}")
        return body

    def wait(self, handle: Mapping[str, Any], *, label: str = "") -> dict[str, Any]:
        """Poll until the request completes, then return the model output."""
        deadline = self._clock() + self.timeout_seconds
        last_status = ""
        while True:
            response = self._get(handle["status_url"])
            if response.status_code >= 400:
                raise FalError(
                    f"fal.ai status check failed (HTTP {response.status_code}): "
                    f"{_error_detail(response, self.api_key)}"
                )
            status_body = response.json()
            status = str(status_body.get("status", ""))
            if status != last_status:
                position = status_body.get("queue_position")
                extra = f", queue position {position}" if position is not None else ""
                logger.info(f"fal.ai {label or handle['request_id']}: {status}{extra}")
                last_status = status
            if status == "COMPLETED":
                if status_body.get("error"):
                    raise FalError(
                        f"fal.ai request {handle['request_id']} failed "
                        f"({status_body.get('error_type', 'error')}): "
                        f"{_redact(status_body['error'], self.api_key)}"
                    )
                break
            if self._clock() >= deadline:
                raise FalError(
                    f"fal.ai request {handle['request_id']} did not finish within "
                    f"{self.timeout_seconds:.0f}s; it may still complete and be billed"
                )
            self._sleep(self.poll_seconds)

        result = self._get(handle["response_url"])
        if result.status_code >= 400:
            raise FalError(
                f"fal.ai result fetch failed (HTTP {result.status_code}): "
                f"{_error_detail(result, self.api_key)}"
            )
        return result.json()

    def run(self, endpoint: str, payload: Mapping[str, Any], *, label: str = "") -> dict[str, Any]:
        return self.wait(self.submit(endpoint, payload), label=label)

    def download(self, url: str, path: str) -> str:
        """Download a fal media URL to ``path`` atomically."""
        if not str(url).startswith(("https://", "http://")):
            raise FalError(f"fal.ai returned an invalid media URL: {url!r}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temp_path = f"{path}.part"
        last_error: Exception | None = None
        for attempt in range(_READ_RETRIES + 1):
            try:
                # Media URLs are public CDN links; don't send the API key there.
                with self.session.get(url, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as response:
                    if response.status_code in _RETRYABLE_STATUS:
                        raise FalError(f"HTTP {response.status_code}")
                    response.raise_for_status()
                    with open(temp_path, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                handle.write(chunk)
                if os.path.getsize(temp_path) == 0:
                    raise FalError("downloaded file is empty")
                os.replace(temp_path, path)
                return path
            except (requests.RequestException, FalError, OSError) as exc:
                last_error = exc
                if attempt < _READ_RETRIES:
                    self._sleep(min(2**attempt, 20))
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise FalError(f"fal.ai media download failed: {last_error}")
