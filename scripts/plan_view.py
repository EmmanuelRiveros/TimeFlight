"""Top-down plan of an era scene for review: proxies by kind, camera path, modern landmarks.

  python scripts/plan_view.py places/tenochtitlan_zocalo/place.json --era 1519 \
      --out docs/img/era_1519_plan.png --half 2200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from timeflight.geo.enu import Origin, geodetic_to_enu

REPO = Path(__file__).resolve().parents[1]
COLORS = {"land": (176, 160, 130), "house": (232, 226, 210), "temple": (205, 160, 110), "shrine": (180, 70, 60),
          "stairs": (215, 185, 140), "wall": (120, 110, 95), "palace": (245, 235, 200), "causeway": (150, 135, 110),
          "bridge": (110, 80, 50), "garden": (110, 150, 80)}
WATER = (70, 110, 125)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("place", type=Path)
    p.add_argument("--era", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--half", type=float, default=2200.0, help="half width of the view, m")
    p.add_argument("--size", type=int, default=1600)
    args = p.parse_args()

    place = json.loads(args.place.read_text(encoding="utf-8"))
    scene = REPO / "data" / "places" / place["id"] / "scene"
    era_dir = scene / f"era_{args.era}"
    era = json.loads((era_dir / "era.json").read_text())
    b = np.load(era_dir / "buildings.npz")
    cam = json.loads((scene / "camera.json").read_text())
    origin = Origin(**cam["origin"])

    S, H = args.size, args.half

    def px(xy):
        xy = np.atleast_2d(xy)
        return [(float(S / 2 + x * S / (2 * H)), float(S / 2 - y * S / (2 * H))) for x, y in xy]

    img = Image.new("RGB", (S, S), WATER)
    d = ImageDraw.Draw(img)
    order = np.argsort([b["base_z"][i] + b["height"][i] for i in range(len(b["height"]))], kind="stable")
    starts = np.concatenate([[0], np.cumsum(b["ring_len"])[:-1]])
    for i in order:
        ring = b["ring_xy"][starts[i]:starts[i] + b["ring_len"][i]]
        if np.abs(ring).min(axis=0).max() > H * 1.2:
            continue
        d.polygon(px(ring), fill=COLORS[era["kinds"][b["kind"][i]]])

    path = np.array(cam["position"])[:, :2]
    d.line(px(path), fill=(220, 30, 30), width=4)
    d.ellipse([*(np.array(px(path[0])[0]) - 8), *(np.array(px(path[0])[0]) + 8)], outline=(220, 30, 30), width=3)

    font = ImageFont.load_default(size=22)
    lm = json.loads((REPO / "data" / "places" / place["id"] / "landmarks.json").read_text(encoding="utf-8"))
    seen = set()
    for l in lm["landmarks"]:
        name = l["name"].split(" de la Ciudad")[0]
        if name in seen:
            continue
        seen.add(name)
        x, y = px(geodetic_to_enu(l["lat"], l["lon"], origin.h, origin)[:2])[0]
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(20, 20, 160))
        d.text((x + 8, y - 12), f"today: {name}", fill=(20, 20, 160), font=font)
    d.text((16, 16), f"{era['label']} (illustrative proxies). Red: camera path (circle = start). "
           f"Blue: modern landmarks (OSM). {2 * H / 1000:.1f} km wide, north up.", fill=(255, 255, 255), font=font)
    d.text((16, S - 34), "© OpenStreetMap contributors (landmark positions)", fill=(255, 255, 255), font=font)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
