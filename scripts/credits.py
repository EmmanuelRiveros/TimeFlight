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
from PIL import Image, ImageDraw, ImageFont

from timeflight.assets_log import read_assets, render_markdown
from timeflight.provenance.credits import build_credits, to_markdown, to_srt
from timeflight.video import probe, read_video, write_video

REPO = Path(__file__).resolve().parents[1]
FONTS = REPO / "data" / "raw" / "fonts"  # Source Sans 3 (OFL-1.1), logged in assets.jsonl
BG, FG, DIM, ACCENT = (20, 18, 16), (236, 230, 218), (170, 162, 148), (255, 214, 150)


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS / f"SourceSans3-{weight}.ttf"
    if not path.exists():
        raise SystemExit(f"{path} missing: download it (see assets.jsonl, kind 'font')")
    return ImageFont.truetype(str(path), size)


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def render_cards(title: str, notice: list[str], sections, size, draft: bool) -> list[Image.Image]:
    """Paginate the notice and the card sections onto as many cards as needed."""
    w, h = size
    s = h / 576
    margin, f_title, f_head, f_body = int(56 * s), font("Semibold", int(30 * s)), font("Semibold", int(19 * s)), \
        font("Regular", int(16 * s))
    gap = int(6 * s)
    blocks = [("title", title)] + [("body", p) for p in notice] + \
             [item for heading, lines in sections for item in [("head", heading)] + [("body", l) for l in lines]]
    cards, img, d, y = [], None, None, 0

    def new_card():
        nonlocal img, d, y
        img = Image.new("RGB", size, BG)
        d = ImageDraw.Draw(img)
        y = margin
        if draft:
            d.text((w - margin, margin // 2), "DRAFT", font=f_head, fill=(230, 80, 60), anchor="rt")
        cards.append(img)

    fonts = {"title": f_title, "head": f_head, "body": f_body}

    def height(kind, text):
        f = fonts[kind]
        return len(wrap(d, text, f, w - 2 * margin)) * int(f.size * 1.25) + (int(14 * s) if kind == "head" else gap)

    new_card()
    for j, (kind, text) in enumerate(blocks):
        f = fonts[kind]
        lines = wrap(d, text, f, w - 2 * margin)
        line_h = int(f.size * 1.25)
        need = height(kind, text)
        if kind == "head" and j + 1 < len(blocks):
            need += height(*blocks[j + 1])  # keep a heading with its first line
        if y + need > h - margin and y > margin:
            new_card()
        if kind == "head":
            y += int(8 * s)
        for line in lines:
            d.text((margin, y), line, font=f, fill={"title": FG, "head": ACCENT, "body": DIM}[kind])
            y += line_h
        y += gap
    return cards


def burn_caption(frame: np.ndarray, text: str, alpha: float, f) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y = int(img.width * 0.05), int(img.height * 0.88)
    a = int(255 * alpha)
    d.text((x + 2, y + 2), text, font=f, fill=(0, 0, 0, int(a * 0.6)), anchor="ls")
    d.text((x, y), text, font=f, fill=(*FG, a), anchor="ls")
    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))


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
    c = build_credits(manifest, assets, "Mexico City")
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

    frames = read_video(args.hero).copy()
    f_cap = font("Semibold", int(size[1] * 0.045))
    fade = 0.4
    for t0, t1, text in captions:
        for i in range(int(t0 * fps), min(int(t1 * fps) + 1, len(frames))):
            t = i / fps
            a = min(1.0, (t - t0) / fade, (t1 - t) / fade)
            if a > 0:
                frames[i] = burn_caption(frames[i], text, a, f_cap)
    hold = int(args.card_seconds * fps)
    ramp = int(0.5 * fps)
    tail = []
    for card in cards:
        arr = np.asarray(card)
        for k in range(hold):
            a = min(1.0, (k + 1) / ramp, (hold - k) / ramp)
            tail.append((np.asarray(BG, np.float32) * (1 - a) + arr * a + 0.5).astype(np.uint8))
    out = args.out_dir / "hero_credits.mp4"
    write_video(np.concatenate([frames, np.stack(tail)]), out, fps, crf=12)
    print(f"wrote {out} ({len(frames) / fps:.1f} s clip + {len(tail) / fps:.1f} s credits)")


if __name__ == "__main__":
    main()
