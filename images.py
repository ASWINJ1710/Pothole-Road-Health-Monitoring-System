from datetime import datetime, timezone

import requests

import geo

SV_URL = "https://maps.googleapis.com/maps/api/streetview"


def fetch_streetview(lat, lon, heading, key):
    """Returns (image_bytes, 'YYYY-MM') or None. Metadata call is free and checks coverage."""
    meta = requests.get(SV_URL + "/metadata", params={"location": f"{lat},{lon}", "key": key}, timeout=15).json()
    if meta.get("status") != "OK":
        return None
    r = requests.get(
        SV_URL,
        params={"size": "640x640", "location": f"{lat},{lon}", "heading": heading,
                "pitch": -15, "fov": 90, "key": key},
        timeout=20,
    )
    return (r.content, meta.get("date")) if r.status_code == 200 else None


def fetch_mapillary(lat, lon, token, radius=0.0005):
    """Returns (image_bytes, iso_date) of the nearest flat photo, or None."""
    r = requests.get(
        "https://graph.mapillary.com/images",
        params={"access_token": token,
                "fields": "id,thumb_1024_url,captured_at,computed_geometry,is_pano",
                "bbox": f"{lon - radius},{lat - radius},{lon + radius},{lat + radius}",
                "limit": 10},
        timeout=20,
    )
    if r.status_code != 200:
        print("Mapillary error", r.status_code, r.text[:200])
        return None
    items = [d for d in r.json().get("data", []) if d.get("thumb_1024_url") and not d.get("is_pano")]
    if not items:
        return None

    def dist(d):
        c = (d.get("computed_geometry") or {}).get("coordinates")
        return geo.hav((lat, lon), (c[1], c[0])) if c else 1e9

    best = min(items, key=dist)
    img = requests.get(best["thumb_1024_url"], timeout=30).content
    ts = best.get("captured_at")
    date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat() if ts else None
    return img, date