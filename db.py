import json
import sqlite3
from contextlib import contextmanager

import pandas as pd

import geo
from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  osm_id TEXT UNIQUE,
  name TEXT,
  coords TEXT NOT NULL,
  length_m REAL,
  images_checked INTEGER DEFAULT 0,
  health REAL
);
CREATE TABLE IF NOT EXISTS detections(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  segment_id INTEGER REFERENCES segments(id),
  lat REAL NOT NULL,
  lon REAL NOT NULL,
  confidence REAL,
  area_ratio REAL,
  severity TEXT,
  source TEXT,
  image_path TEXT,
  captured_at TEXT,
  detected_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_det_seg ON detections(segment_id);
CREATE TABLE IF NOT EXISTS scans(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  source TEXT,
  south REAL, west REAL, north REAL, east REAL,
  roads INTEGER DEFAULT 0,
  photos INTEGER DEFAULT 0,
  potholes INTEGER DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS analysed_images(
  image_key TEXT PRIMARY KEY,
  scan_id INTEGER,
  n_potholes INTEGER
);
"""


@contextmanager
def conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)


def clear():
    with conn() as c:
        c.execute("DELETE FROM detections")
        c.execute("DELETE FROM segments")
        c.execute("DELETE FROM scans")
        c.execute("DELETE FROM analysed_images")


def clear_demo():
    """Remove only the synthetic demo roads and potholes."""
    with conn() as c:
        c.execute("DELETE FROM detections WHERE source='demo' OR segment_id IN "
                  "(SELECT id FROM segments WHERE osm_id LIKE 'demo_%')")
        c.execute("DELETE FROM segments WHERE osm_id LIKE 'demo_%'")


def add_scan(name, source, bbox):
    with conn() as c:
        cur = c.execute("INSERT INTO scans(name,source,south,west,north,east) VALUES(?,?,?,?,?,?)",
                        (name, source, *bbox))
        return cur.lastrowid


def finish_scan(scan_id, roads, photos, potholes):
    with conn() as c:
        c.execute("UPDATE scans SET roads=?, photos=?, potholes=? WHERE id=?", (roads, photos, potholes, scan_id))


def scans_df():
    with conn() as c:
        return pd.read_sql_query("SELECT * FROM scans ORDER BY id DESC", c)


def image_seen(key):
    with conn() as c:
        return c.execute("SELECT 1 FROM analysed_images WHERE image_key=?", (key,)).fetchone() is not None


def mark_image(key, scan_id, n_potholes):
    with conn() as c:
        c.execute("INSERT OR IGNORE INTO analysed_images(image_key,scan_id,n_potholes) VALUES(?,?,?)",
                  (key, scan_id, n_potholes))


def add_segment(osm_id, name, coords, length_m):
    with conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO segments(osm_id,name,coords,length_m) VALUES(?,?,?,?)",
            (osm_id, name, json.dumps(coords), length_m),
        )
        return c.execute("SELECT id FROM segments WHERE osm_id=?", (osm_id,)).fetchone()[0]


def bump_checked(segment_id):
    if segment_id is None:
        return
    with conn() as c:
        c.execute("UPDATE segments SET images_checked=images_checked+1 WHERE id=?", (segment_id,))


def add_detection(d):
    with conn() as c:
        c.execute(
            "INSERT INTO detections(segment_id,lat,lon,confidence,area_ratio,severity,source,image_path,captured_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                d.get("segment_id"), d["lat"], d["lon"], d.get("confidence"), d.get("area_ratio"),
                d.get("severity"), d.get("source"), d.get("image_path"), d.get("captured_at"),
            ),
        )


def set_health_many(rows):
    with conn() as c:
        c.executemany("UPDATE segments SET health=? WHERE id=?", rows)


def segments_df():
    sql = """SELECT s.id, s.name, s.coords, s.length_m, s.health, s.images_checked,
                    COUNT(d.id) AS n_potholes
             FROM segments s LEFT JOIN detections d ON d.segment_id = s.id
             GROUP BY s.id"""
    with conn() as c:
        df = pd.read_sql_query(sql, c)
    df["coords"] = df["coords"].apply(json.loads)
    df["health"] = pd.to_numeric(df["health"], errors="coerce")
    return df


def detections_df():
    with conn() as c:
        return pd.read_sql_query("SELECT * FROM detections", c)


def nearest_segment(lat, lon, max_m=40):
    best, best_d = None, max_m
    for s in segments_df().itertuples():
        for p in s.coords:
            d = geo.hav((lat, lon), p)
            if d < best_d:
                best, best_d = s.id, d
    return best