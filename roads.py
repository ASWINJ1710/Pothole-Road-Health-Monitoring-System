import time
import xml.etree.ElementTree as ET

import requests

OSM_API = "https://api.openstreetmap.org/api/0.6/map"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
HEADERS = {
    "User-Agent": "pothole-road-health/1.0 (student project)",
    "Accept": "application/json",
}
ROAD_TYPES = {"motorway", "trunk", "primary", "secondary", "tertiary",
              "unclassified", "residential", "service"}


def bbox_too_big(bbox, max_deg=0.03):
    s, w, n, e = bbox
    return (n - s) > max_deg or (e - w) > max_deg


def _from_osm_api(bbox):
    """Main OpenStreetMap server. Works for small areas (limit about 50,000 map points)."""
    s, w, n, e = bbox
    r = requests.get(OSM_API, params={"bbox": f"{w},{s},{e},{n}"}, headers={"User-Agent": HEADERS["User-Agent"]},
                     timeout=(10, 60))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {r.text[:80].strip()}")
    root = ET.fromstring(r.content)
    nodes = {nd.attrib["id"]: (float(nd.attrib["lat"]), float(nd.attrib["lon"])) for nd in root.iter("node")}
    ways = []
    for wy in root.iter("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in wy.iter("tag")}
        if tags.get("highway") not in ROAD_TYPES:
            continue
        coords = [nodes[nd.attrib["ref"]] for nd in wy.iter("nd") if nd.attrib["ref"] in nodes]
        if len(coords) > 1:
            ways.append({"id": int(wy.attrib["id"]), "name": tags.get("name", "Unnamed road"), "coords": coords})
    return ways


def _from_overpass(bbox, errors):
    s, w, n, e = bbox
    q = (
        '[out:json][timeout:30];'
        'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service)$"]'
        f"({s},{w},{n},{e});out geom;"
    )
    for url in OVERPASS_URLS:
        host = url.split("/")[2]
        try:
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=(8, 40))
            if r.status_code != 200:
                errors.append(f"{host}: HTTP {r.status_code}")
                continue
            ways = []
            for el in r.json().get("elements", []):
                g = el.get("geometry")
                if g and len(g) > 1:
                    ways.append({"id": el["id"], "name": el.get("tags", {}).get("name", "Unnamed road"),
                                 "coords": [(p["lat"], p["lon"]) for p in g]})
            return ways
        except Exception as err:
            errors.append(f"{host}: {type(err).__name__}")
    return None


def fetch_roads(bbox):
    """bbox = (south, west, north, east). Returns list of ways with coords [(lat, lon), ...]."""
    errors = []
    try:
        return _from_osm_api(bbox)
    except Exception as err:
        errors.append(f"api.openstreetmap.org: {err}")
    for attempt in range(2):
        ways = _from_overpass(bbox, errors)
        if ways is not None:
            return ways
        time.sleep(3)
    raise RuntimeError("Could not load roads. " + "; ".join(errors))
