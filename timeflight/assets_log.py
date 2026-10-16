"""Single place to record every external asset (data, weights, images, voices).

Entries live in `assets.jsonl` at the repo root (committed: it is metadata, not data).
`render_markdown()` turns them into ASSETS.md. An asset without an entry must not be used.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO_ROOT / "assets.jsonl"

# Sources that must never appear in this project (see CLAUDE.md).
FORBIDDEN_HOSTS = (
    "google.com",
    "googleapis.com",
    "gstatic.com",
    "cesium.com",
    "arcgis.com",
    "esri.com",
    "mapbox.com",
    "virtualearth.net",
    "bing.com",
    "davidrumsey.com",
    "mapillary.com",
)


@dataclass(frozen=True)
class Asset:
    name: str
    kind: str  # "dem" | "osm" | "image" | "model" | "voice" | "text" | ...
    source_url: str
    license: str
    attribution: str
    retrieved: str = field(default_factory=lambda: _dt.date.today().isoformat())
    notes: str = ""


def check_source(url: str) -> None:
    """Raise if the URL points to a forbidden provider."""
    lowered = url.lower()
    for host in FORBIDDEN_HOSTS:
        if host in lowered:
            raise ValueError(f"Forbidden source ({host}): {url}")


def read_assets(log_path: Path = DEFAULT_LOG) -> list[Asset]:
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as f:
        return [Asset(**json.loads(line)) for line in f if line.strip()]


def log_asset(
    name: str,
    kind: str,
    source_url: str,
    license: str,
    attribution: str,
    notes: str = "",
    log_path: Path = DEFAULT_LOG,
) -> Asset:
    """Append an asset entry unless (name, source_url) is already logged."""
    check_source(source_url)
    if not license.strip():
        raise ValueError(f"Asset {name!r} has no license")
    for existing in read_assets(log_path):
        if existing.name == name and existing.source_url == source_url:
            return existing
    asset = Asset(name, kind, source_url, license, attribution, notes=notes)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(asset), ensure_ascii=False) + "\n")
    return asset


def render_markdown(assets: list[Asset]) -> str:
    lines = [
        "# Assets",
        "",
        "Generated from `assets.jsonl` by `timeflight.assets_log`. Do not edit by hand.",
        "",
        "| Name | Kind | Source | License | Attribution | Retrieved | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in sorted(assets, key=lambda a: (a.kind, a.name)):
        cells = [a.name, a.kind, a.source_url, a.license, a.attribution, a.retrieved, a.notes]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    (REPO_ROOT / "ASSETS.md").write_text(render_markdown(read_assets()), encoding="utf-8")
