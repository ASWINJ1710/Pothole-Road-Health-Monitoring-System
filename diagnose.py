"""Runs every stage of the area scan in the terminal and says which one fails.
Usage: python diagnose.py            (uses the latest saved scan area, or a default area)
       python diagnose.py S W N E    (your own area, e.g. python diagnose.py 10.985 76.425 10.990 76.435)"""
import sys

import config
import db
import geo
import roads
import autoscan


def step(name):
    print(f"\n--- {name}")


db.init()
if len(sys.argv) == 5:
    bbox = tuple(float(x) for x in sys.argv[1:])
else:
    scans = db.scans_df()
    bbox = (scans.iloc[0].south, scans.iloc[0].west, scans.iloc[0].north, scans.iloc[0].east) if not scans.empty \
        else (10.985, 76.425, 10.990, 76.435)
print("Area (south, west, north, east):", bbox)
print("Database file:", config.DB_PATH)
print("Model file exists:", config.MODEL_PATH.exists(), "|", config.MODEL_PATH)
print("Mapillary token set:", bool(config.MAPILLARY_TOKEN), "| starts with:", config.MAPILLARY_TOKEN[:4])

step("1. Roads")
try:
    ways = roads.fetch_roads(bbox)
    print("OK:", len(ways), "roads")
except Exception as e:
    sys.exit(f"FAILED: {e}")

pieces, points = [], []
for wy in ways:
    for i, ch in enumerate(geo.chunk_way(wy["coords"], 100)):
        pieces.append((f"{wy['id']}_{i}", wy["name"], ch))
for pi, (_, _, ch) in enumerate(pieces):
    points += [(pi, la, lo, h) for la, lo, h in geo.interpolate(ch, 20)]
print("segments:", len(pieces), "| sample points:", len(points))

step("2. Mapillary photos in the area")
try:
    photos = autoscan.fetch_mapillary_area(bbox, config.MAPILLARY_TOKEN)
    print("OK:", len(photos), "usable photos")
except Exception as e:
    sys.exit(f"FAILED: {e}")

step("3. Photos within 30 m of a road point")
jobs = autoscan.match_photos(points, photos)
print("OK:", len(jobs), "matches")
if not jobs:
    sys.exit("No photos on these roads. This area has no usable Mapillary coverage.")

step("4. Download the first photo")
try:
    got = autoscan.fetch_photo(False, config.MAPILLARY_TOKEN, *jobs[0][1:4], jobs[0][4])
    print("OK:", len(got[0]), "bytes")
except Exception as e:
    sys.exit(f"FAILED: {type(e).__name__}: {e}")

step("5. Run the model on it")
try:
    import detect
    dets, _ = detect.run(got[0], 0.35)
    print("OK:", len(dets), "pothole(s) found in this photo")
except Exception as e:
    sys.exit(f"FAILED: {type(e).__name__}: {e}")

print("\nAll stages passed. A scan of this area should work.")