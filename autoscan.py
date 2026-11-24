
"""Scan tab: find roads, collect Mapillary/Street View images and detect potholes."""

import io
import math
from datetime import datetime, timezone

import numpy as np
import folium
import requests
import streamlit as st
from PIL import Image
from folium.plugins import Draw
from streamlit_folium import st_folium

import config
import db
import detect
import geo
import images
import roads
import score

HEADERS = {"User-Agent": "pothole-road-health/1.0 (student project)"}
MAPILLARY_API = "https://graph.mapillary.com/images"


def geocode(query):
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 1},
        headers=HEADERS,
        timeout=15,
    )
    data = response.json() if response.status_code == 200 else []
    return (
        (float(data[0]["lat"]), float(data[0]["lon"]))
        if data else None
    )


def fetch_mapillary_area(bbox, token, limit=2000):
    """Fetch Mapillary images, including panoramas."""
    south, west, north, east = bbox

    response = requests.get(
        MAPILLARY_API,
        params={
            "access_token": token,
            "fields": (
                "id,thumb_1024_url,captured_at,computed_geometry,"
                "is_pano,compass_angle"
            ),
            "bbox": f"{west},{south},{east},{north}",
            "limit": min(limit, 2000),
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Mapillary returned {response.status_code}: "
            f"{response.text[:300]}"
        )

    photos = []

    for item in response.json().get("data", []):
        coordinates = (
            item.get("computed_geometry") or {}
        ).get("coordinates")

        url = item.get("thumb_1024_url")

        if not coordinates or not url:
            continue

        timestamp = item.get("captured_at")

        captured_at = (
            datetime.fromtimestamp(
                timestamp / 1000,
                tz=timezone.utc,
            ).isoformat()
            if timestamp else None
        )

        photos.append({
            "id": str(item["id"]),
            "lat": float(coordinates[1]),
            "lon": float(coordinates[0]),
            "url": url,
            "captured_at": captured_at,
            "is_pano": bool(item.get("is_pano")),
            "compass_angle": float(item.get("compass_angle") or 0),
        })

    return photos


def match_photos(points, photos, max_m=30):
    """Match each road sample point with its nearest unused photo."""
    jobs = []
    used = set()

    for pi, lat, lon, heading in points:
        best_index = None
        best_distance = max_m

        for index, photo in enumerate(photos):
            if index in used:
                continue

            if abs(photo["lat"] - lat) > 0.0004:
                continue

            if abs(photo["lon"] - lon) > 0.0004:
                continue

            distance = geo.hav(
                (lat, lon),
                (photo["lat"], photo["lon"]),
            )

            if distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index is not None:
            used.add(best_index)
            photo = photos[best_index]

            jobs.append((
                pi,
                photo["lat"],
                photo["lon"],
                heading,
                photo,
            ))

    return jobs


def image_key(use_sv, lat, lon, heading, photo):
    if use_sv:
        return f"sv:{lat:.5f},{lon:.5f},{int(heading // 45)}"

    return f"mly:{photo['id']}"


def fetch_photo(use_sv, key, lat, lon, heading, photo):
    """Download an image and return (bytes, captured_at)."""
    if use_sv:
        return images.fetch_streetview(
            lat, lon, heading, key
        )

    response = requests.get(
        photo["url"],
        timeout=30,
        headers=HEADERS,
    )
    response.raise_for_status()

    return response.content, photo["captured_at"]


def panorama_views(image_bytes, road_heading, compass_angle):
    """
    Convert an equirectangular 360-degree panorama into
    two perspective views facing along and opposite the road.

    Returns a list of JPEG image bytes.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    source = np.asarray(image)
    height, width = source.shape[:2]

    output_width = 640
    output_height = 480
    horizontal_fov = 90
    vertical_fov = 75

    views = []

    # View in both directions along the road.
    for target_heading in (
        road_heading,
        road_heading + 180,
    ):
        relative_heading = (
            target_heading - compass_angle + 180
        ) % 360 - 180

        yaw = math.radians(relative_heading)
        half_fov = math.radians(horizontal_fov / 2)
        vertical_half = math.radians(vertical_fov / 2)

        x_angles = np.linspace(
            -half_fov, half_fov, output_width
        )
        y_angles = np.linspace(
            vertical_half, -vertical_half, output_height
        )

        xx, yy = np.meshgrid(x_angles, y_angles)

        # Perspective camera rays.
        ray_x = np.tan(xx)
        ray_y = np.tan(yy)
        ray_z = np.ones_like(ray_x)

        norm = np.sqrt(
            ray_x ** 2 + ray_y ** 2 + ray_z ** 2
        )

        ray_x /= norm
        ray_y /= norm
        ray_z /= norm

        # Rotate the camera around the vertical axis.
        world_x = ray_x * math.cos(yaw) + ray_z * math.sin(yaw)
        world_z = -ray_x * math.sin(yaw) + ray_z * math.cos(yaw)

        longitude = np.arctan2(world_x, world_z)
        latitude = np.arcsin(
            np.clip(ray_y, -1.0, 1.0)
        )

        # Equirectangular pixel coordinates.
        map_x = (
            (longitude / (2 * math.pi) + 0.5) * width
        ) % width

        map_y = (
            (0.5 - latitude / math.pi) * height
        )

        map_x = np.clip(map_x, 0, width - 1).astype(np.int32)
        map_y = np.clip(map_y, 0, height - 1).astype(np.int32)

        view = source[map_y, map_x]
        output = Image.fromarray(view)

        buffer = io.BytesIO()
        output.save(buffer, format="JPEG", quality=92)
        views.append(buffer.getvalue())

    return views


def analyse_one_image(
    pieces,
    pi,
    lat,
    lon,
    heading,
    photo,
    use_sv,
    key,
    conf,
    scan_id,
    stats,
    job_number,
):
    """Download and analyse one original image."""
    image_id = image_key(
        use_sv, lat, lon, heading, photo
    )

    if db.image_seen(image_id):
        stats["skipped"] += 1
        return

    try:
        downloaded = fetch_photo(
            use_sv, key, lat, lon, heading, photo
        )

        if downloaded is None:
            raise RuntimeError("No image available")

        image_bytes, captured_at = downloaded

        if use_sv:
            views = [image_bytes]
        elif photo.get("is_pano"):
            views = panorama_views(
                image_bytes,
                heading,
                photo.get("compass_angle", 0),
            )
        else:
            views = [image_bytes]

        all_detections = []
        annotated_images = []

        for view_number, view_bytes in enumerate(views):
            detections, annotated = detect.run(
                view_bytes, conf
            )

            if detections:
                all_detections.extend(detections)
                annotated_images.append(annotated)

        oid, name, coordinates = pieces[pi]

        segment_id = db.add_segment(
            oid,
            name,
            coordinates,
            geo.path_length(coordinates),
        )

        db.bump_checked(segment_id)
        stats["analysed"] += 1

        saved_path = None

        if annotated_images:
            # Save the first annotated view containing detections.
            saved_path = str(
                config.IMG_DIR
                / f"{segment_id}_{job_number}.jpg"
            )

            annotated_images[0].save(saved_path)

        for detection in all_detections:
            db.add_detection({
                "segment_id": segment_id,
                "lat": lat,
                "lon": lon,
                "confidence": detection["confidence"],
                "area_ratio": detection["area_ratio"],
                "severity": detection["severity"],
                "source": (
                    "streetview" if use_sv else "mapillary"
                ),
                "image_path": saved_path,
                "captured_at": captured_at,
            })

            stats["found"] += 1

        # Mark the original photo only after successful analysis.
        db.mark_image(
            image_id,
            scan_id,
            len(all_detections),
        )

    except Exception as error:
        stats["failed"] += 1

        if stats["first_error"] is None:
            stats["first_error"] = (
                f"{type(error).__name__}: {error}"
            )


def analyse_jobs(
    pieces,
    jobs,
    use_sv,
    key,
    conf,
    scan_id,
    progress=None,
):
    """Analyse original photos and track progress."""
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)

    stats = {
        "analysed": 0,
        "found": 0,
        "failed": 0,
        "skipped": 0,
        "first_error": None,
    }

    total = len(jobs)

    for index, (
        pi, lat, lon, heading, photo
    ) in enumerate(jobs):

        analyse_one_image(
            pieces,
            pi,
            lat,
            lon,
            heading,
            photo,
            use_sv,
            key,
            conf,
            scan_id,
            stats,
            index,
        )

        if progress and total:
            progress((index + 1) / total)

    score.recompute()
    return stats


@st.fragment
def render(conf):
    msg = st.session_state.get("scan_msg")

    if msg:
        (st.success if msg[0] == "success" else st.error)(
            msg[1]
        )

    st.write(
        "**1.** Search a place (optional)  "
        "**2.** Draw a rectangle around the roads you want to check  "
        "**3.** Press Scan."
    )

    query = st.text_input(
        "Search a place",
        placeholder="Town, area or road name",
    )

    if query and query != st.session_state.get("last_place"):
        st.session_state["last_place"] = query

        try:
            location = geocode(query)
        except Exception:
            location = None

        if location:
            st.session_state["center"] = location
        else:
            st.warning(
                "Place not found. Try a bigger town name."
            )

    center = st.session_state.get(
        "center", (20.59, 78.96)
    )

    zoom = 16 if "center" in st.session_state else 5

    map_key = f"map_obj_{center}"

    if map_key not in st.session_state:
        fmap = folium.Map(
            location=list(center),
            zoom_start=zoom,
        )

        Draw(
            export=False,
            draw_options={
                "rectangle": True,
                "polyline": False,
                "polygon": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
            },
        ).add_to(fmap)

        st.session_state[map_key] = fmap

    fmap = st.session_state[map_key]

    output = st_folium(
        fmap,
        height=450,
        use_container_width=True,
        key=f"area_map_{center}",
        returned_objects=["last_active_drawing"],
    )

    bbox = None
    drawing = (output or {}).get("last_active_drawing")

    if drawing:
        ring = drawing["geometry"]["coordinates"][0]
        longitudes = [point[0] for point in ring]
        latitudes = [point[1] for point in ring]

        bbox = (
            min(latitudes),
            min(longitudes),
            max(latitudes),
            max(longitudes),
        )

    with st.expander("Or enter a bounding box manually"):
        a, b, c, d = st.columns(4)

        south = a.number_input(
            "South", value=0.0, format="%.6f", key="as_s"
        )
        west = b.number_input(
            "West", value=0.0, format="%.6f", key="as_w"
        )
        north = c.number_input(
            "North", value=0.0, format="%.6f", key="as_n"
        )
        east = d.number_input(
            "East", value=0.0, format="%.6f", key="as_e"
        )

        if north > south and east > west:
            bbox = (south, west, north, east)

    if bbox:
        mid = (bbox[0] + bbox[2]) / 2

        st.caption(
            f"Marked area: "
            f"{(bbox[3] - bbox[1]) * 111 * math.cos(math.radians(mid)):.1f} km wide x "
            f"{(bbox[2] - bbox[0]) * 111:.1f} km high"
        )

    scan_name = st.text_input(
        "Name this scan",
        value=st.session_state.get("last_place") or "Area scan",
        key="as_name",
    )

    # Safer default: keep previous results.
    fresh = st.checkbox(
        "Start fresh: delete all older saved data before this scan",
        value=False,
        key="as_fresh",
    )

    col1, col2, col3 = st.columns(3)

    source = col1.selectbox(
        "Photo source",
        ["Mapillary (free)", "Google Street View (paid)"],
        key="as_src",
    )

    step = col2.slider(
        "Sample every (m)", 10, 50, 20, 5, key="as_step"
    )

    max_images = col3.slider(
        "Max photos", 10, 300, 100, 10, key="as_max"
    )

    use_sv = source.startswith("Google")

    api_key = (
        config.GOOGLE_API_KEY
        if use_sv else config.MAPILLARY_TOKEN
    )

    problems = []

    if bbox is None:
        problems.append("Draw an area on the map first.")
    elif roads.bbox_too_big(bbox):
        problems.append(
            "The area is too large. Draw a smaller rectangle."
        )

    if not api_key:
        problems.append(
            "Missing API key in .env."
        )

    if not config.MODEL_PATH.exists():
        problems.append("Missing models/best.pt.")

    for problem in problems:
        st.warning(problem)

    if not st.button(
        "Scan marked area",
        type="primary",
        disabled=bool(problems),
        key="as_go",
    ):
        return

    st.session_state.pop("scan_msg", None)

    with st.status(
        "Scanning the marked area...",
        expanded=True,
    ) as status:

        st.write("Finding roads in OpenStreetMap...")

        try:
            ways = roads.fetch_roads(bbox)
        except Exception as error:
            status.update(
                label="Could not load roads",
                state="error",
            )
            st.error(str(error))
            return

        if not ways:
            status.update(
                label="No roads found",
                state="error",
            )
            return

        pieces = []
        points = []

        for way in ways:
            chunks = geo.chunk_way(
                way["coords"],
                config.SEGMENT_LEN_M,
            )

            for index, coordinates in enumerate(chunks):
                pieces.append((
                    f"{way['id']}_{index}",
                    way["name"],
                    coordinates,
                ))

        for piece_index, (_, _, coordinates) in enumerate(pieces):
            points.extend([
                (piece_index, lat, lon, heading)
                for lat, lon, heading in geo.interpolate(
                    coordinates, step
                )
            ])

        st.write(
            f"{len(ways)} roads, {len(pieces)} segments, "
            f"{len(points)} sample points."
        )

        if use_sv:
            jobs = [
                (pi, lat, lon, heading, None)
                for pi, lat, lon, heading in points
            ]
        else:
            st.write(
                "Looking for Mapillary photos, including panoramas..."
            )

            try:
                photos = fetch_mapillary_area(
                    bbox, api_key
                )
            except Exception as error:
                status.update(
                    label="Mapillary request failed",
                    state="error",
                )
                st.error(str(error))
                return

            panorama_count = sum(
                photo["is_pano"] for photo in photos
            )

            st.write(
                f"Found {len(photos)} Mapillary photos: "
                f"{panorama_count} panoramas and "
                f"{len(photos) - panorama_count} normal images."
            )

            jobs = match_photos(points, photos)

            st.write(
                f"{len(jobs)} road points matched with photos."
            )

        if not jobs:
            status.update(
                label="No matching street photos",
                state="error",
            )
            st.error(
                "No Mapillary images were found close to the selected "
                "roads. Try a different area or use Google Street View."
            )
            return

        if len(jobs) > max_images:
            interval = math.ceil(len(jobs) / max_images)
            jobs = jobs[::interval]

        st.write(
            f"Testing the first image and YOLO model..."
        )

        tested = False
        last_error = "No image could be downloaded."

        for pi, lat, lon, heading, photo in jobs[:5]:
            try:
                downloaded = fetch_photo(
                    use_sv,
                    api_key,
                    lat,
                    lon,
                    heading,
                    photo,
                )

                if downloaded is None:
                    continue

                image_bytes = downloaded[0]

                if not use_sv and photo.get("is_pano"):
                    views = panorama_views(
                        image_bytes,
                        heading,
                        photo.get("compass_angle", 0),
                    )
                    detect.run(views[0], conf)
                else:
                    detect.run(image_bytes, conf)

                tested = True
                break

            except Exception as error:
                last_error = (
                    f"{type(error).__name__}: {error}"
                )

        if not tested:
            status.update(
                label="Test failed. No scan data changed.",
                state="error",
            )
            st.error(last_error)
            return

        # Create scan before analysing so images receive the correct ID.
        if fresh:
            db.clear()

        scan_id = db.add_scan(
            scan_name or "Area scan",
            "streetview" if use_sv else "mapillary",
            bbox,
        )

        st.write(
            f"Test passed. Analysing {len(jobs)} original photos..."
        )

        progress_bar = st.progress(0.0)

        stats = analyse_jobs(
            pieces,
            jobs,
            use_sv,
            api_key,
            conf,
            scan_id,
            progress_bar.progress,
        )

        db.finish_scan(
            scan_id,
            len(ways),
            stats["analysed"],
            stats["found"],
        )

        score.recompute()

        if stats["analysed"] == 0:
            status.update(
                label="No new photos were analysed",
                state="error",
            )

            st.session_state["scan_msg"] = (
                "error",
                "No new photos were analysed. "
                f"Skipped: {stats['skipped']}, "
                f"failed: {stats['failed']}. "
                f"First error: {stats['first_error']}",
            )

            st.rerun()

        status.update(
            label="Scan finished and saved",
            state="complete",
        )

    st.session_state.pop("dash_area", None)

    extra = (
        f" First error: {stats['first_error']}"
        if stats["failed"] and stats["first_error"]
        else ""
    )

    st.session_state["scan_msg"] = (
        "success",
        f"Saved scan '{scan_name or 'Area scan'}': "
        f"{stats['analysed']} original photos analysed, "
        f"{stats['skipped']} previously analysed, "
        f"{stats['found']} potholes found, "
        f"{stats['failed']} failed.{extra} "
        "Open the Dashboard tab.",
    )

    st.rerun()
