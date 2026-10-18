"""Credit cards and burned-in captions (PIL, imported lazily; the rest of timeflight stays numpy-only).

Fonts: Source Sans 3 (SIL OFL 1.1) in data/raw/fonts, fetched and logged by scripts/fetch_fonts.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FONTS = REPO / "data" / "raw" / "fonts"  # Source Sans 3 (OFL-1.1), logged in assets.jsonl
BG, FG, DIM, ACCENT = (20, 18, 16), (236, 230, 218), (170, 162, 148), (255, 214, 150)


def font(weight: str, size: int):
    from PIL import ImageFont

    path = FONTS / f"SourceSans3-{weight}.ttf"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/fetch_fonts.py")
    return ImageFont.truetype(str(path), size)


def wrap(draw, text: str, f, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def render_cards(title: str, notice: list[str], sections, size, draft: bool) -> list:
    """Paginate the notice and the card sections onto as many PIL images as needed."""
    from PIL import Image, ImageDraw

    w, h = size
    s = min(w, h) / 576  # same type size on a 1024x576 card and a 576x1024 one
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
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y = int(img.width * 0.05), int(img.height * 0.88)
    a = int(255 * alpha)
    lines = wrap(d, text, f, int(img.width * 0.9))
    line_h = int(f.size * 1.2)
    for k, line in enumerate(lines):  # wrap upward from the baseline, so the last line sits at y
        ly = y - (len(lines) - 1 - k) * line_h
        d.text((x + 2, ly + 2), line, font=f, fill=(0, 0, 0, int(a * 0.6)), anchor="ls")
        d.text((x, ly), line, font=f, fill=(*FG, a), anchor="ls")
    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))


def caption_frames(frames: np.ndarray, captions, fps: float, fade: float = 0.4) -> np.ndarray:
    """Burn (start_s, end_s, text) captions into a copy of uint8 frames (F, H, W, 3), fading in and out."""
    out = frames.copy()
    f_cap = font("Semibold", int(min(frames.shape[1:3]) * 0.045))
    for t0, t1, text in captions:
        for i in range(int(t0 * fps), min(int(t1 * fps) + 1, len(out))):
            t = i / fps
            a = min(1.0, (t - t0) / fade, (t1 - t) / fade)
            if a > 0:
                out[i] = burn_caption(out[i], text, a, f_cap)
    return out


def card_frames(cards, fps: float, seconds: float, ramp_s: float = 0.5) -> np.ndarray:
    """Hold each card for `seconds`, fading from and to the background color."""
    hold, ramp = int(seconds * fps), max(int(ramp_s * fps), 1)
    tail = []
    for card in cards:
        arr = np.asarray(card)
        for k in range(hold):
            a = min(1.0, (k + 1) / ramp, (hold - k) / ramp)
            tail.append((np.asarray(BG, np.float32) * (1 - a) + arr * a + 0.5).astype(np.uint8))
    return np.stack(tail)
