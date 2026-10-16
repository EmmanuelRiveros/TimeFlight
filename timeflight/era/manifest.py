"""Era manifest: loading and validation (provenance rules from CLAUDE.md).

Every element must cite at least one listed source and state a confidence. Sources must be
public domain / CC0 / CC BY images or texts, or "facts only" citations.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIDENCE = ("high", "medium", "low", "illustrative")
ALLOWED_LICENSE_WORDS = ("public domain", "no copyright", "no known restrictions", "cc0", "cc by ",
                         "cc by-4", "facts only", "odbl")
FORBIDDEN_LICENSE_WORDS = ("nc", "non-commercial", "noncommercial", "all rights reserved", "sa ")
ELEMENT_KEYS = ("anchor", "remove", "water", "island", "canals", "houses", "precinct")
ELEMENT_LISTS = ("plazas", "structures", "causeways")


def license_ok(text: str) -> bool:
    t = f" {text.lower()} "
    if any(f" {w}" in t or f"-{w}" in t for w in FORBIDDEN_LICENSE_WORDS):
        return False
    return any(w in t for w in ALLOWED_LICENSE_WORDS)


def elements(manifest: dict):
    for key in ELEMENT_KEYS:
        if key in manifest:
            yield key, manifest[key]
    for key in ELEMENT_LISTS:
        for item in manifest.get(key, []):
            yield f"{key}:{item.get('name', '?')}", item


def validate(manifest: dict) -> list[str]:
    problems = []
    for field in ("era", "label", "year", "sources", "anchor"):
        if field not in manifest:
            problems.append(f"missing {field}")
    ids = set()
    for s in manifest.get("sources", []):
        for f in ("id", "title", "url", "license", "credit"):
            if not s.get(f):
                problems.append(f"source {s.get('id', '?')}: missing {f}")
        if s.get("license") and not license_ok(s["license"]):
            problems.append(f"source {s.get('id')}: license not allowed: {s['license']!r}")
        ids.add(s.get("id"))
    for name, el in elements(manifest):
        srcs = el.get("sources") or []
        if not srcs:
            problems.append(f"{name}: no sources")
        for s in srcs:
            if s not in ids:
                problems.append(f"{name}: unknown source {s}")
        if el.get("confidence") not in CONFIDENCE:
            problems.append(f"{name}: confidence must be one of {CONFIDENCE}")
    return problems


def load(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate(manifest)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return manifest
