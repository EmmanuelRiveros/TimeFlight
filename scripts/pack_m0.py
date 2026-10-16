"""Zip control videos for upload to Drive (only mp4 + json, no PNG frames).

  .venv/Scripts/python scripts/pack_m0.py                                   # -> out/m0_inputs.zip
  .venv/Scripts/python scripts/pack_m0.py --src out/m1 --folders r1024_361 --out out/m1_inputs.zip
"""

import argparse
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "out"

p = argparse.ArgumentParser()
p.add_argument("--src", type=Path, default=OUT / "m0")
p.add_argument("--folders", nargs="+", default=["r576_49", "r1024_121", "r1024_241"])
p.add_argument("--out", type=Path, default=OUT / "m0_inputs.zip")
args = p.parse_args()

with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
    for folder in args.folders:
        for f in sorted((args.src / folder).glob("*")):
            if f.suffix in {".mp4", ".json"}:
                z.write(f, f"{folder}/{f.name}")
                print(f"{folder}/{f.name}")
print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
