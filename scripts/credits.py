"""Credits from the era manifest and assets.jsonl: markdown, credit cards, captions, and the hero
with captions burned in and the cards appended.

  .venv/Scripts/python scripts/credits.py --place places/tenochtitlan_zocalo/place.json --era 1519 \
      --hero out/m4/timewave_v4.mp4 --out-dir out/m4/credits

Writes places/<id>/credits_<era>.md (committed), ASSETS.md, and in --out-dir: card_NN.png,
captions.srt and hero_credits.mp4. Refuses a final render while the credits are a draft
(an unverified license, an invalid manifest) unless --allow-draft, which stamps every card DRAFT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from timeflight.assets_log import read_assets, render_markdown
from timeflight.provenance.cards import card_frames, caption_frames, render_cards
from timeflight.provenance.credits import build_credits, to_markdown, to_srt
from timeflight.video import probe, read_video, write_video

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--place", type=Path, required=True)
    p.add_argument("--era", required=True)
    p.add_argument("--hero", type=Path, help="time-wave clip to caption and append the cards to")
    p.add_argument("--out-dir", type=Path, default=REPO / "out" / "m4" / "credits")
    p.add_argument("--present-caption", default="Mexico City, today")
    p.add_argument("--past-caption", default=None, help="default: '<era label> · illustrative AI reconstruction'")
    p.add_argument("--present-span", type=float, nargs=2, default=[0.5, 3.3], help="caption start/end, s")
    p.add_argument("--past-span", type=float, nargs=2, default=[12.0, 15.0])
    p.add_argument("--card-seconds", type=float, default=6.0)
    p.add_argument("--allow-draft", action="store_true")
    args = p.parse_args()

    place = json.loads(args.place.read_text(encoding="utf-8"))
    manifest = json.loads((args.place.parent / "eras" / f"{args.era}.json").read_text(encoding="utf-8"))
    assets = read_assets()
    c = build_credits(manifest, assets, place.get("label"))
    md_path = args.place.parent / f"credits_{args.era}.md"
    md_path.write_text(to_markdown(c), encoding="utf-8")
    (REPO / "ASSETS.md").write_text(render_markdown(assets), encoding="utf-8")
    print(f"wrote {md_path} and ASSETS.md")
    if c.draft:
        print("credits are a DRAFT:\n  " + "\n  ".join(c.problems))
        if not args.allow_draft:
            raise SystemExit("fix the problems above or pass --allow-draft")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    past_caption = args.past_caption or f"{manifest['label']} · illustrative AI reconstruction"
    captions = [(*args.present_span, args.present_caption), (*args.past_span, past_caption)]
    (args.out_dir / "captions.srt").write_text(to_srt(captions), encoding="utf-8")

    info = probe(args.hero) if args.hero else {"width": 1024, "height": 576, "fps": 24.0}
    size, fps = (info["width"], info["height"]), info["fps"]
    cards = render_cards(c.title, c.notice, c.cards, size, c.draft)
    for i, card in enumerate(cards):
        card.save(args.out_dir / f"card_{i:02d}.png")
    print(f"wrote {len(cards)} cards to {args.out_dir}")
    if not args.hero:
        return

    frames = caption_frames(read_video(args.hero), captions, fps)
    tail = card_frames(cards, fps, args.card_seconds)
    out = args.out_dir / "hero_credits.mp4"
    write_video(np.concatenate([frames, tail]), out, fps, crf=12)
    print(f"wrote {out} ({len(frames) / fps:.1f} s clip + {len(tail) / fps:.1f} s credits)")


if __name__ == "__main__":
    main()
