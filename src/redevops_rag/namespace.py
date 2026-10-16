"""Optional namespace guard.

Tenant scoping is the caller's responsibility: a store or graph built without a namespace performs no
isolation. A component constructed with ``namespace=...`` rejects any request that names a different
namespace (or none), so a mix-up fails loudly instead of returning another corpus's data.
"""
from __future__ import annotations

from typing import Optional


class NamespaceMismatch(PermissionError):
    """The request's namespace does not match the namespace the component was constructed with."""


def check_namespace(owner: Optional[str], requested: Optional[str]) -> None:
    """No-op when ``owner`` is None (unscoped). Otherwise ``requested`` must equal ``owner``."""
    if owner is not None and requested != owner:
        raise NamespaceMismatch("request namespace does not match the namespace of this component")
