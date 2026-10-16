"""Download and cut the open geodata for one place: Copernicus GLO-30 DEM and OSM buildings.

  python scripts/fetch_geodata.py places/tenochtitlan_zocalo/place.json

Writes to data/places/<id>/ (gitignored; OSM-derived files fall under ODbL and are not committed):
  dem.npz          heights (orthometric, m) + geotransform, cropped to the place's DEM box
  buildings.json   OSM building footprints (ENU meters around the origin) with heights
  landmarks.json   named OSM features listed in place.json, with their centroids
Raw downloads are cached in data/raw/. Every source is logged in assets.jsonl.
Needs the `geo` extra (rasterio, osmium).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from timeflight.assets_log import log_asset
from timeflight.geo.dem import Dem
from timeflight.geo.enu import Origin
from timeflight.geo.osm import footprint

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
UA = {"User-Agent": "TimeFlight/0.1 (+https://github.com/EmmanuelRiveros/TimeFlight)"}

DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_LICENSE = "Copernicus WorldDEM-30 licence (free; reproduction, distribution, adaptation)"
DEM_LICENSE_URL = ("https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/"
                   "resources/license/License-COPDEM-30.pdf")
DEM_CREDIT = ("produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and "
              "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights "
              "reserved. The organisations in charge of the Copernicus programme by law or by delegation "
              "do not incur any liability for any use of the Copernicus WorldDEM-30.")
OSM_CREDIT = "© OpenStreetMap contributors (ODbL 1.0), extract by Geofabrik GmbH"


def download(url: str, path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"download {url}", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(path)
    return path


def dem_tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def fetch_dem(box: dict, out: Path) -> None:
    import rasterio
    from rasterio.merge import merge

    files = []
    for lat in range(int(np.floor(box["south"])), int(np.floor(box["north"])) + 1):
        for lon in range(int(np.floor(box["west"])), int(np.floor(box["east"])) + 1):
            name = dem_tile_name(lat, lon)
            url = f"{DEM_BUCKET}/{name}/{name}.tif"
            files.append(download(url, RAW / "dem" / f"{name}.tif"))
            log_asset(name=name, kind="dem", source_url=url, license=f"{DEM_LICENSE} {DEM_LICENSE_URL}",
                      attribution=DEM_CREDIT, notes="Copernicus DEM GLO-30 Public via AWS Open Data "
                      "(registry.opendata.aws/copernicus-dem). DSM: includes buildings and trees.")
    srcs = [rasterio.open(f) for f in files]
    heights, transform = merge(srcs, bounds=(box["west"], box["south"], box["east"], box["north"]))
    for s in srcs:
        s.close()
    np.savez_compressed(out / "dem.npz", heights=heights[0].astype(np.float32),
                        lon0=transform.c, dlon=transform.a, lat0=transform.f, dlat=transform.e)
    print(f"dem.npz {heights.shape[1:]} px, {heights.min():.0f}..{heights.max():.0f} m", flush=True)


def fetch_pbf(url: str) -> Path:
    pbf = download(url, RAW / "osm" / url.rsplit("/", 1)[1])
    md5 = download(url + ".md5", RAW / "osm" / (pbf.name + ".md5")).read_text().split()[0]
    h = hashlib.md5()
    with pbf.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    if h.hexdigest() != md5:
        pbf.unlink()
        raise SystemExit(f"MD5 mismatch for {pbf.name}; deleted, run again")
    log_asset(name=pbf.name, kind="osm", source_url=url, license="ODbL 1.0 (opendatacommons.org/licenses/odbl/1-0/)",
              attribution=OSM_CREDIT, notes=f"md5 {md5}. Derived data stays in data/ and is not committed.")
    return pbf


def extract_osm(pbf: Path, box: dict, origin: Origin, landmark_names: list[str]) -> tuple[list, list]:
    """Building areas (ways and multipolygons) whose first outer node is inside the box, and
    areas whose name contains one of `landmark_names` (case-insensitive)."""
    import osmium

    def inside(lat, lon):
        return box["south"] <= lat <= box["north"] and box["west"] <= lon <= box["east"]

    buildings, landmarks = [], []
    t0 = time.time()
    fp = osmium.FileProcessor(str(pbf)).with_locations().with_areas()
    for obj in fp:
        if not isinstance(obj, osmium.osm.Area):
            continue
        tags = dict(obj.tags)
        name = tags.get("name", "")
        is_landmark = bool(name) and any(s.lower() in name.lower() for s in landmark_names)
        is_building = "building" in tags or "building:part" in tags
        if not is_building and not is_landmark:
            continue
        rings = [[(n.lat, n.lon) for n in ring] for ring in obj.outer_rings()]
        if not rings or not inside(*rings[0][0]):
            continue
        osm_id = f"{'w' if obj.from_way() else 'r'}{obj.orig_id()}"
        if is_landmark:
            pts = np.array([p for r in rings for p in r])
            landmarks.append({"id": osm_id, "name": name, "lat": float(pts[:, 0].mean()),
                              "lon": float(pts[:, 1].mean()), "tags": tags})
        if is_building:
            for k, ring in enumerate(rings):
                fpr = footprint(f"{osm_id}_{k}" if len(rings) > 1 else osm_id, tags, ring, origin)
                if fpr:
                    buildings.append(fpr)
    print(f"osm: {len(buildings)} building footprints, {len(landmarks)} landmarks "
          f"in {time.time() - t0:.0f} s", flush=True)
    return buildings, landmarks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("place", type=Path)
    p.add_argument("--skip-dem", action="store_true")
    p.add_argument("--skip-osm", action="store_true")
    args = p.parse_args()

    place = json.loads(args.place.read_text(encoding="utf-8"))
    out = REPO / "data" / "places" / place["id"]
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_dem:
        fetch_dem(place["dem_box"], out)
    d = np.load(out / "dem.npz")
    dem = Dem(d["heights"], float(d["lon0"]), float(d["dlon"]), float(d["lat0"]), float(d["dlat"]))
    origin = Origin(place["origin"]["lat"], place["origin"]["lon"],
                    float(dem.sample(place["origin"]["lat"], place["origin"]["lon"])))
    if not args.skip_osm:
        pbf = fetch_pbf(place["osm"]["pbf_url"])
        buildings, landmarks = extract_osm(pbf, place["osm"]["box"], origin, place["osm"].get("landmarks", []))
        meta = {"origin": {**place["origin"], "h": origin.h}, "source": place["osm"]["pbf_url"], "credit": OSM_CREDIT,
                "license": "ODbL 1.0: do not commit this file"}
        (out / "buildings.json").write_text(json.dumps({**meta, "buildings": buildings}), encoding="utf-8")
        (out / "landmarks.json").write_text(json.dumps({**meta, "landmarks": landmarks}, indent=1,
                                                       ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
