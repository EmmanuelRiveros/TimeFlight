"""Zip the M0 control videos for upload to Drive (only mp4 + json, no PNG frames).

  .venv/Scripts/python scripts/pack_m0.py   # -> out/m0_inputs.zip
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "out"
FOLDERS = ["r512_49", "r960_121", "r960_241"]

with zipfile.ZipFile(ROOT / "m0_inputs.zip", "w", zipfile.ZIP_STORED) as z:
    for folder in FOLDERS:
        for f in sorted((ROOT / "m0" / folder).glob("*")):
            if f.suffix in {".mp4", ".json"}:
                z.write(f, f"{folder}/{f.name}")
                print(f"{folder}/{f.name}")
print(f"wrote {ROOT / 'm0_inputs.zip'} ({(ROOT / 'm0_inputs.zip').stat().st_size / 1e6:.1f} MB)")
