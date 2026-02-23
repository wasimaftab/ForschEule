"""Shared HTTP helpers with retries and exponential backoff."""

from __future__ import annotations

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_RETRY_STATUSES = (429, 500, 502, 503, 504)
_MAX_RETRIES = 4
_BACKOFF_FACTOR = 1.0  # 1s, 2s, 4s, 8s


def make_session() -> requests.Session:
    """Create a requests session with retry/backoff on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=list(_RETRY_STATUSES),
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def rate_limited_get(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    delay: float = 0.0,
    timeout: float = 30.0,
) -> requests.Response:
    """GET with pre-request delay for rate limiting."""
    if delay > 0:
        time.sleep(delay)
    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp
