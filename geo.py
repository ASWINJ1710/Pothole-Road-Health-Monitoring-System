import math

R = 6371000.0


def hav(p, q):
    la1, lo1, la2, lo2 = map(math.radians, (p[0], p[1], q[0], q[1]))
    a = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing(p, q):
    la1, la2 = math.radians(p[0]), math.radians(q[0])
    dlo = math.radians(q[1] - p[1])
    y = math.sin(dlo) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def path_length(coords):
    return sum(hav(p, q) for p, q in zip(coords, coords[1:]))


def chunk_way(coords, chunk_m=100):
    """Split a polyline into pieces of roughly chunk_m metres."""
    chunks, cur, acc = [], [coords[0]], 0.0
    for p, q in zip(coords, coords[1:]):
        acc += hav(p, q)
        cur.append(q)
        if acc >= chunk_m:
            chunks.append(cur)
            cur, acc = [q], 0.0
    if len(cur) > 1:
        chunks.append(cur)
    return chunks


def interpolate(coords, step_m=20):
    """Points every step_m metres along a polyline: (lat, lon, heading)."""
    pts, carry = [], 0.0
    for p, q in zip(coords, coords[1:]):
        d = hav(p, q)
        if d == 0:
            continue
        h = bearing(p, q)
        t = carry
        while t <= d:
            f = t / d
            pts.append((p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f, h))
            t += step_m
        carry = t - d
    return pts
