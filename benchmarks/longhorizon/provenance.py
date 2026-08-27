"""Implementation provenance for every benchmark result cell.

A frozen-control A/B is only trustworthy if each result can prove *which code produced it*. The failure we
are guarding against is subtle and real: a result labelled "F4 enterprise" while the harness was actually
still running the AGPL skeleton (an enterprise-arm commit that never reached main). CI metadata does not
catch that — the artifact itself must carry its provenance.

So every cell is stamped with:
  • the git commit + dirty flag of each repo whose code ran (harness=redevops-rag, contextos, CR-enterprise);
  • a content sha256 of each active arm's implementation file — the strongest signal, independent of git
    state: it changes the moment the arm's source changes, committed or not;
  • the candidate/plan fingerprint schema versions (plan-cache key schema, spec version) — so a result can
    never silently compare across incompatible arm-identity or key-shape schemas.

A reviewer (or a diff against the frozen control) can then reject any cell whose arm source hash, enterprise
commit, or key schema does not match what the claim assumes. Provenance is part of the result, not a side
channel.
"""
from __future__ import annotations

import hashlib
import subprocess
from importlib import import_module
from pathlib import Path


def _git(repo: Path | None, *args) -> str | None:
    if repo is None:
        return None
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _repo_of(path: Path | None) -> Path | None:
    if path is None:
        return None
    top = _git(path.parent, "rev-parse", "--show-toplevel")
    return Path(top) if top else None


def _module_file(modname: str) -> Path | None:
    try:
        return Path(import_module(modname).__file__)   # type: ignore[arg-type]
    except Exception:
        return None


def _sha256(path: Path | None) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path else None
    except Exception:
        return None


def _repo_prov(repo: Path | None, *scope: Path) -> dict | None:
    if repo is None:
        return None
    commit = _git(repo, "rev-parse", "HEAD")
    porcelain = _git(repo, "status", "--porcelain", *[str(s) for s in scope])
    return {"commit": commit, "dirty": bool(porcelain) if porcelain is not None else None}


def _schema() -> dict:
    out: dict = {}
    try:
        from context_runtime.plancache.cache import KEY_SCHEMA
        out["plan_cache_key"] = KEY_SCHEMA
    except Exception:
        out["plan_cache_key"] = None
    try:
        from context_runtime.types import SPEC_VERSION
        out["spec_version"] = SPEC_VERSION
    except Exception:
        out["spec_version"] = None
    return out


# The implementation file behind each arm — the content hash of this file *is* the arm's identity.
_ARM_IMPL = {
    "cr-enterprise": ("sparse-regions", "context_runtime_enterprise.sparse_regions"),
    "cr-materialize": ("materialization", "context_runtime_enterprise.materialization"),
}


def capture(arms_active: list[str]) -> dict:
    """Provenance for a run using ``arms_active``. Enterprise repo/arm entries appear only when an enterprise
    arm actually ran, so an AGPL-only run is not mislabelled as carrying enterprise code."""
    harness_dir = Path(__file__).resolve().parent
    prov: dict = {
        "harness": _repo_prov(_repo_of(harness_dir), harness_dir),
        "contextos": _repo_prov(_repo_of(_module_file("context_runtime.types"))),
        "schema": _schema(),
        "arms": {},
    }
    if any(a in _ARM_IMPL for a in arms_active):
        prov["enterprise"] = _repo_prov(_repo_of(_module_file("context_runtime_enterprise")))
    for arm in arms_active:
        if arm in _ARM_IMPL:
            impl, modname = _ARM_IMPL[arm]
            prov["arms"][arm] = {"impl": impl, "src_sha256": _sha256(_module_file(modname))}
    return prov


def one_line(prov: dict) -> str:
    def short(d):
        if not d or not d.get("commit"):
            return "?"
        return d["commit"][:10] + ("*" if d.get("dirty") else "")
    parts = [f"harness={short(prov.get('harness'))}", f"contextos={short(prov.get('contextos'))}"]
    if "enterprise" in prov:
        parts.append(f"enterprise={short(prov.get('enterprise'))}")
    parts.append(f"key_schema={prov['schema'].get('plan_cache_key')}")
    for arm, a in prov.get("arms", {}).items():
        parts.append(f"{arm}={a.get('src_sha256')}")
    return " ".join(parts)
