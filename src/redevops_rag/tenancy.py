"""Tenant-aware mode: an explicit, fail-closed scoping contract.

Two modes, chosen when a store is constructed:

* **unscoped** (``tenant_aware=False``, the default) — exactly the historical behaviour; every existing single-user
  and local call keeps working with no identity. An unscoped store provides *no* isolation and must never be reported
  as evidence of it (``RetrievalResult.isolation == "unscoped"``).
* **tenant-aware** (``tenant_aware=True``) — every read, write and admin call must carry a valid ``tenant_scope``;
  a missing or invalid one is refused (:class:`TenantScopeError`, or ``DENIED`` from ``hybrid_search_result``) and never
  falls back to an unrestricted search. Rows carry their scope in a ``tenant_scope`` column, every retrieval leg filters on
  it in the ``WHERE`` clause *before* ``ORDER BY ... LIMIT``, and chunk ids are qualified with the scope so one tenant cannot
  overwrite another's row or prune another's documents.

This module only validates and qualifies identifiers. It does **not** authenticate anyone: the caller (a gateway, the
Context Runtime control plane, an API layer) must establish the identity and pass the verified scope in.
"""
from __future__ import annotations

import unicodedata
from typing import Any

MAX_SCOPE_LEN = 256
_BAD_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}
_RESERVED = {"*", "%", "all", "any", "none", "null", "public"}     # values that read as "everyone" or "nobody"


class TenantScopeError(PermissionError):
    """A tenant scope was required and missing, or was supplied and invalid. The request is refused."""


def validate_tenant_scope(scope: Any) -> str:
    """Return ``scope`` if it is a usable tenant identifier, else raise :class:`TenantScopeError`.

    Refused: non-strings, empty/blank, leading/trailing whitespace, control characters (incl. NUL), over-long values,
    and wildcard-like sentinels. The message never echoes the offending value."""
    if not isinstance(scope, str):
        raise TenantScopeError(f"tenant scope must be a string, got {type(scope).__name__}")
    if not scope or not scope.strip():
        raise TenantScopeError("tenant scope is empty")
    if scope != scope.strip():
        raise TenantScopeError("tenant scope has leading or trailing whitespace")
    if len(scope) > MAX_SCOPE_LEN:
        raise TenantScopeError(f"tenant scope longer than {MAX_SCOPE_LEN} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in scope):
        raise TenantScopeError("tenant scope contains control characters")
    if any(unicodedata.category(ch) in _BAD_CATEGORIES for ch in scope):
        raise TenantScopeError("tenant scope contains format, surrogate, private-use or unassigned characters")
    if any(unicodedata.category(ch) == "Zs" and ch != " " for ch in scope):
        raise TenantScopeError("tenant scope contains a non-ASCII space separator")
    if unicodedata.normalize("NFC", scope) != scope:     # look-alike spellings of the same name must not be separate tenants
        raise TenantScopeError("tenant scope is not Unicode NFC-normalised")
    if "::" in scope:       # the id separator: scope "a::b" + id "c" would qualify to the same id as scope "a" + id "b::c"
        raise TenantScopeError("tenant scope must not contain '::'")
    if scope.lower() in _RESERVED:
        raise TenantScopeError("tenant scope is a reserved or wildcard-like value")
    return scope


def qualify_id(scope: str, chunk_id: str) -> str:
    """The stored id of ``chunk_id`` inside a tenant-aware store: tenants can never collide on an id."""
    return f"{scope}::{chunk_id}"
