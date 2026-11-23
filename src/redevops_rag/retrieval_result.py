"""Explicit retrieval outcomes.

A search that returns ``[]`` can mean "nothing matches" or "the backend is down". Callers (and benchmarks) must be
able to tell them apart, so the new :func:`redevops_rag.retrieve.hybrid_search_result` returns a
:class:`RetrievalResult` whose :class:`RetrievalStatus` is one of

* ``COMPLETE``    every leg ran and at least one hit came back;
* ``EMPTY``       every leg ran and nothing matched (a genuine no-match, or a blank query);
* ``DEGRADED``    at least one leg failed or was unavailable but another ran — the hits are usable but incomplete,
  and an empty hit list does **not** mean "no match";
* ``UNAVAILABLE`` no leg could run — the empty hit list means nothing;
* ``DENIED``      the request was refused by the namespace / tenant guard before any retrieval ran.

Error text is passed through :func:`redact` so a connection string or password never reaches a log or artifact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class RetrievalStatus(str, Enum):
    COMPLETE = "complete"
    EMPTY = "empty"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DENIED = "denied"


class BackendUnavailable(RuntimeError):
    """A store could not run a search leg (extension not loaded, index missing, connection lost, query error)."""


_URL_CRED = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s:@]*:[^/\s@]*@")
_KV_CRED = re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key)\s*[=:]\s*\S+")


def redact(message: str, limit: int = 240) -> str:
    """Strip credentials from an error message and bound its length."""
    msg = _URL_CRED.sub(r"\1***@", str(message))
    msg = _KV_CRED.sub(lambda m: f"{m.group(1)}=***", msg)
    msg = " ".join(msg.split())
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


@dataclass(frozen=True)
class LegReport:
    """What one retrieval leg did. ``ok`` is False when the leg could not run; ``n_hits`` is then 0 and meaningless."""
    name: str
    ok: bool
    n_hits: int = 0
    error: Optional[str] = None
    checked: bool = True          # False: the store cannot report failures for this leg, so ok=True is unverified


@dataclass
class RetrievalResult:
    hits: list[dict[str, Any]]
    status: RetrievalStatus
    legs: tuple[LegReport, ...] = ()
    as_of: Optional[datetime] = None        # the "now" used for recency scoring (wall clock when not supplied)
    notes: list[str] = field(default_factory=list)
    # How the request was isolated: "tenant_aware" (the store enforces a tenant scope and one was applied) or
    # "unscoped" (no enforcement — NEVER evidence of tenant isolation, whatever the data happened to contain).
    isolation: str = "unscoped"

    @property
    def usable(self) -> bool:
        """True when ``hits`` may be used as retrieved evidence (COMPLETE, DEGRADED or a genuine EMPTY)."""
        return self.status in (RetrievalStatus.COMPLETE, RetrievalStatus.DEGRADED, RetrievalStatus.EMPTY)

    @property
    def complete_coverage(self) -> bool:
        """True only when every leg ran (COMPLETE or EMPTY) — i.e. an empty list really means no match."""
        return self.status in (RetrievalStatus.COMPLETE, RetrievalStatus.EMPTY)

    def failed_legs(self) -> list[LegReport]:
        return [leg for leg in self.legs if not leg.ok]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "n_hits": len(self.hits), "as_of": self.as_of.isoformat() if self.as_of else None, "isolation": self.isolation,
                "legs": [{"name": l.name, "ok": l.ok, "n_hits": l.n_hits, "error": l.error, "checked": l.checked} for l in self.legs],
                "notes": list(self.notes)}


class RetrievalUnavailable(RuntimeError):
    """Raised by the legacy list-returning :func:`hybrid_search` when no leg could run (so ``[]`` would have lied)."""

    def __init__(self, result: RetrievalResult):
        self.result = result
        detail = "; ".join(f"{l.name}: {l.error}" for l in result.failed_legs()) or "no retrieval leg could run"
        super().__init__(f"retrieval unavailable ({detail})")
