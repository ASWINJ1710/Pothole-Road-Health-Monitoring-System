"""Video tab: run the pothole detector on a road video and save results with GPS positions."""
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import cv2
import streamlit as st

import config
import db
import detect
import geo
import score

MERGE_M = 8        # detections closer than this (metres along the route) count as one pothole
MAX_FRAMES = 400   # safety limit so long videos do not take forever


def parse_gpx(data):
    pts = []
    for el in ET.fromstring(data).iter():
        if el.tag.split("}")[-1] in ("trkpt", "rtept"):
            pts.append((float(el.attrib["lat"]), float(el.attrib["lon"])))
    return pts


def densify(route, step_m=25):
    pts = [(la, lo) for la, lo, _ in geo.interpolate(route, step_m)]
    if not pts or pts[-1] != route[-1]:
        pts.append(route[-1])
    return pts


def point_at(route, d):
    acc = 0.0
    for p, q in zip(route, route[1:]):
        seg = geo.hav(p, q)
        if seg > 0 and acc + seg >= d:
            f = (d - acc) / seg
            return p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f
        acc += seg
    return route[-1]


def process_video(path, route, name, conf, every_s=1.0, progress=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, int(round(fps * every_s)), math.ceil(n_frames / MAX_FRAMES))

    dense = densify(route)
    total = geo.path_length(dense)
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    sids, ends, acc = [], [], 0.0
    for i, ch in enumerate(geo.chunk_way(dense, config.SEGMENT_LEN_M)):
        length = geo.path_length(ch)
        acc += length
        sids.append(db.add_segment(f"video_{run_id}_{i}", name, ch, length))
        ends.append(acc)

    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    captured = datetime.now(timezone.utc).isoformat()
    stats = {"frames": 0, "frames_with": 0, "saved": 0, "segments": len(sids), "gallery": []}
    last_saved, frame_no = -1e9, 0

    while cap.grab():
        if frame_no % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                d = total * (frame_no / n_frames)
                idx = next((i for i, e in enumerate(ends) if d <= e), len(sids) - 1)
                lat, lon = point_at(dense, d)
                if frame.shape[1] > 1280:
                    frame = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
                _, buf = cv2.imencode(".jpg", frame)
                dets, annotated = detect.run(buf.tobytes(), conf)
                db.bump_checked(sids[idx])
                stats["frames"] += 1
                if dets:
                    stats["frames_with"] += 1
                    if d - last_saved >= MERGE_M:
                        last_saved = d
                        img_path = str(config.IMG_DIR / f"video_{run_id}_{frame_no}.jpg")
                        annotated.save(img_path)
                        for dt in dets:
                            db.add_detection({
                                "segment_id": sids[idx], "lat": lat, "lon": lon,
                                "confidence": dt["confidence"], "area_ratio": dt["area_ratio"],
                                "severity": dt["severity"], "source": "video",
                                "image_path": img_path, "captured_at": captured,
                            })
                        stats["saved"] += len(dets)
                        stats["gallery"].append((annotated, f"{frame_no / fps:.0f}s: {len(dets)} pothole(s)"))
        frame_no += 1
        if progress:
            progress(min(frame_no / n_frames, 1.0))
    cap.release()
    score.recompute()
    return stats


def render(conf):
    st.write("Upload a video from a car, bike or scooter, with the phone camera pointing at the road ahead.")
    video = st.file_uploader("Road video", type=["mp4", "mov", "avi", "mkv"])
    mode = st.radio("Where was it recorded?", ["Straight road: start and end points", "GPX file"], horizontal=True)

    route = None
    if mode.startswith("Straight"):
        a, b, c, d = st.columns(4)
        lat1 = a.number_input("Start latitude", value=0.0, format="%.6f")
        lon1 = b.number_input("Start longitude", value=0.0, format="%.6f")
        lat2 = c.number_input("End latitude", value=0.0, format="%.6f")
        lon2 = d.number_input("End longitude", value=0.0, format="%.6f")
        if (lat1, lon1) != (0.0, 0.0) and (lat2, lon2) != (0.0, 0.0):
            route = [(lat1, lon1), (lat2, lon2)]
        st.caption("Tip: right-click a spot in Google Maps and click the coordinates to copy them.")
    else:
        gpx = st.file_uploader("GPX file", type=["gpx"])
        if gpx:
            try:
                route = parse_gpx(gpx.getvalue())
            except Exception:
                route = []
            if len(route) < 2:
                st.error("Could not read track points from this GPX file.")
                route = None

    o1, o2 = st.columns(2)
    every = o1.slider("Check a frame every (seconds)", 0.5, 3.0, 1.0, 0.5)
    name = o2.text_input("Road name", "Video route")

    problems = []
    if not video:
        problems.append("Upload a video first.")
    if route is None:
        problems.append("Enter the route (start and end points or a GPX file).")
    elif geo.path_length(route) < 20:
        problems.append("The route is shorter than 20 m. Check the coordinates.")
        route = None
    if not config.MODEL_PATH.exists():
        problems.append("Missing models/best.pt.")
    for p in problems:
        st.warning(p)

    if st.button("Analyse video", type="primary", disabled=bool(problems)):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video.getvalue())
        tmp.close()
        bar = st.progress(0.0)
        try:
            stats = process_video(tmp.name, route, name, conf, every, bar.progress)
        finally:
            os.remove(tmp.name)
        st.success(
            f"{stats['frames']} frames checked, {stats['frames_with']} had potholes, "
            f"{stats['saved']} potholes saved on {stats['segments']} road segments. Open the Dashboard tab."
        )
        cols = st.columns(3)
        for i, (img, cap_text) in enumerate(stats["gallery"][:9]):
            with cols[i % 3]:
                st.image(img, caption=cap_text)