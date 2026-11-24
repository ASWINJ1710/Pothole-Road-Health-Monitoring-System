import pandas as pd

import db

WEIGHTS = {"low": 3, "medium": 7, "high": 15}


def severity(area_ratio):
    if area_ratio < 0.02:
        return "low"
    if area_ratio < 0.06:
        return "medium"
    return "high"


def label(health):
    if health is None or pd.isna(health):
        return "Not scanned"
    if health >= 75:
        return "Good"
    if health >= 50:
        return "Fair"
    return "Poor"


def recompute():
    """health = 100 - weighted pothole penalty per 100 m, clipped to 0..100."""
    seg, det = db.segments_df(), db.detections_df()
    updates = []
    for s in seg.itertuples():
        if s.images_checked == 0:
            updates.append((None, s.id))
            continue
        d = det[det.segment_id == s.id]
        pen = sum(WEIGHTS.get(r.severity, 3) * (r.confidence or 0) for r in d.itertuples())
        pen *= 100.0 / max(s.length_m, 30.0)
        updates.append((max(0.0, 100.0 - pen), s.id))
    db.set_health_many(updates)
