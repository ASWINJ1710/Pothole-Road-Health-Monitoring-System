import random
from datetime import datetime, timedelta, timezone

import db
import geo
import score


def load(center=(12.9716, 77.5946), rows=5, cols=6, seed=7):
    """Synthetic roads + potholes so the dashboard can be tried without a model or API keys."""
    rnd = random.Random(seed)
    db.init()
    db.clear_demo()
    names = ["MG Road", "Station Road", "Park Street", "Market Road", "Temple Road"]
    step_lat, step_lon = 0.0012, 0.0009
    now = datetime.now(timezone.utc)
    for r in range(rows):
        lat0 = center[0] + (r - rows / 2) * step_lat
        badness = rnd.choice([0.1, 0.3, 0.6, 0.9])
        for k in range(cols):
            lon0 = center[1] + (k - cols / 2) * step_lon
            a = (lat0 + rnd.uniform(-2e-5, 2e-5), lon0)
            b = (lat0 + rnd.uniform(-2e-5, 2e-5), lon0 + step_lon)
            sid = db.add_segment(f"demo_{r}_{k}", names[r % len(names)], [a, b], geo.path_length([a, b]))
            for _ in range(rnd.randint(3, 6)):
                db.bump_checked(sid)
            for _ in range(min(10, int(rnd.expovariate(1 / (badness * 5 + 0.2))))):
                f = rnd.random()
                ratio = rnd.choice([0.008, 0.015, 0.03, 0.05, 0.08, 0.12])
                db.add_detection({
                    "segment_id": sid,
                    "lat": a[0] + (b[0] - a[0]) * f + rnd.uniform(-1e-5, 1e-5),
                    "lon": a[1] + (b[1] - a[1]) * f,
                    "confidence": rnd.uniform(0.4, 0.95),
                    "area_ratio": ratio,
                    "severity": score.severity(ratio),
                    "source": "demo",
                    "captured_at": (now - timedelta(days=rnd.randint(0, 700))).isoformat(),
                })
    score.recompute()