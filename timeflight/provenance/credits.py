"""Credits from the era manifest and the assets log (never hand-written).

Every historical source in the manifest is credited with its license and the elements it was
used for, at the stated confidence. Data, models and software come from `assets.jsonl`.
`problems` lists anything that must be fixed before the credits are final (an unverified
license, an invalid manifest); renderers mark such credits as DRAFT.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..assets_log import Asset
from ..era.manifest import elements, validate

UNVERIFIED_WORDS = ("unverified", "to confirm", "verify", "believed", "tbd")
ELEMENT_LABELS = {"anchor": "anchor position", "remove": "removal of modern buildings", "water": "lake",
                  "island": "island shoreline", "canals": "canals and chinampas", "houses": "houses",
                  "precinct": "sacred precinct"}
SECTION_KINDS = (("Map and terrain data", ("dem", "osm")), ("Models", ("model",)),
                 ("Software and fonts", ("software", "font")))


@dataclass
class Credits:
    title: str
    notice: list[str]
    sections: list[tuple[str, list[str]]] = field(default_factory=list)
    cards: list[tuple[str, list[str]]] = field(default_factory=list)  # shorter lines for on-screen cards
    problems: list[str] = field(default_factory=list)

    @property
    def draft(self) -> bool:
        return bool(self.problems)


def _element_label(name: str, el: dict) -> str:
    if ":" in name:
        return el.get("name", name.split(":", 1)[1])
    return ELEMENT_LABELS.get(name, name)


def _evaluation_only(a: Asset) -> bool:
    return "evaluation only" in a.notes.lower()


def build_credits(manifest: dict, assets: list[Asset], place_label: str | None = None) -> Credits:
    label = manifest["label"]
    els = list(elements(manifest))
    conf = Counter(el.get("confidence") for _, el in els)
    notice = [
        f"{label}: an illustrative AI reconstruction, not a historical record.",
        f"The camera follows real coordinates{f' over {place_label}' if place_label else ''}. The era geometry is "
        f"built from the sources below: {conf['high'] + conf['medium']} of {len(els)} elements are placed with high "
        f"or medium confidence, {conf['low'] + conf['illustrative']} are low-confidence or illustrative.",
        "Images were generated with open-weight video models. The LTX-2.x Community License is free only for "
        "entities under $10M annual revenue, so this stack is open-weight, not fully open source.",
    ]
    c = Credits(title=f"{label} - sources and credits", notice=notice)
    c.problems += [f"manifest: {p}" for p in validate(manifest)]

    used: dict[str, list[str]] = {s["id"]: [] for s in manifest.get("sources", [])}
    for name, el in els:
        for sid in el.get("sources") or []:
            used.setdefault(sid, []).append(f"{_element_label(name, el)} ({el.get('confidence')})")
    lines, short = [], []
    for s in manifest.get("sources", []):
        uses = used.get(s["id"]) or []
        if not uses:
            continue  # listed but not used by any element: no credit needed
        lines.append(f"{s['credit']}. {s['title']}. License: {s['license']}. Used for: {', '.join(uses)}.")
        short.append(f"{s['credit']} ({s['license']})")
    c.sections.append(("Historical sources", lines))
    c.cards.append(("Historical sources", short))

    for heading, kinds in SECTION_KINDS:
        seen, lines = set(), []
        for a in sorted((a for a in assets if a.kind in kinds and not _evaluation_only(a)), key=lambda a: a.name):
            text = f"{a.name}: {a.attribution}. License: {a.license}."
            if a.kind in ("osm", "font") or a.attribution.lower().startswith(a.name.lower()):
                text = f"{a.attribution}. License: {a.license}."
            if a.kind == "dem":
                text = f"Terrain {a.attribution} License: Copernicus WorldDEM-30 licence."
            if text not in seen:
                seen.add(text)
                lines.append(text)
        c.sections.append((heading, lines))
        if lines:
            c.cards.append((heading, lines))
    evaluation = [f"{a.name} ({a.license}), used only to measure results" for a in assets if _evaluation_only(a)]
    if evaluation:
        c.sections.append(("Evaluation only", evaluation))

    for a in assets:
        if any(w in a.license.lower() for w in UNVERIFIED_WORDS):
            c.problems.append(f"asset {a.name}: license not verified ({a.license})")
    return c


def to_markdown(c: Credits) -> str:
    out = [f"# {c.title}", ""]
    if c.draft:
        out += ["**DRAFT: fix before publishing:**", ""] + [f"- {p}" for p in c.problems] + [""]
    out += [" ".join(c.notice), ""]
    for heading, lines in c.sections:
        out += [f"## {heading}", ""] + [f"- {line}" for line in lines] + [""]
    return "\n".join(out)


def _ts(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def to_srt(captions: list[tuple[float, float, str]]) -> str:
    """Captions as (start_s, end_s, text) -> SubRip text."""
    blocks = []
    for i, (t0, t1, text) in enumerate(sorted(captions), start=1):
        if t1 <= t0:
            raise ValueError(f"caption {text!r} ends before it starts")
        blocks.append(f"{i}\n{_ts(t0)} --> {_ts(t1)}\n{text}\n")
    return "\n".join(blocks)
