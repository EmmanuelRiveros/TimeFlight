"""Download the credit-card font (Source Sans 3, SIL OFL 1.1) into data/raw/fonts and log it.

  python scripts/fetch_fonts.py
"""

from __future__ import annotations

import urllib.request

from timeflight.assets_log import log_asset
from timeflight.provenance.cards import FONTS

URL = "https://github.com/adobe-fonts/source-sans/raw/release/TTF/SourceSans3-{}.ttf"

if __name__ == "__main__":
    FONTS.mkdir(parents=True, exist_ok=True)
    for weight in ("Regular", "Semibold"):
        path, url = FONTS / f"SourceSans3-{weight}.ttf", URL.format(weight)
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        log_asset(name=path.name, kind="font", source_url=url, license="SIL Open Font License 1.1",
                  attribution="Source Sans 3 by Adobe",
                  notes="Used for credit cards and captions. License read from the repo's LICENSE.md on 2026-10-17.")
        print(f"{path} ({path.stat().st_size} bytes)")
