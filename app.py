from datetime import datetime, timezone

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

import autoscan
import config
import db
import demo
import detect
import score
import video_tab

st.set_page_config(page_title="Pothole & Road Health", page_icon="🛣️", layout="wide")
db.init()

SEV_COLORS = {"low": "#f2c94c", "medium": "#f2994a", "high": "#d63b3b"}
CAT_COLORS = {"Good": "#2e9e4f", "Fair": "#e0a800", "Poor": "#d63b3b", "Not scanned": "#9aa0a6"}


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("Road health")
    st.caption("Pothole detection + road condition analytics")
    conf = st.slider("Detection confidence", 0.10, 0.90, config.CONF_THRES, 0.05)
    if config.MODEL_PATH.exists():
        st.success("Model loaded: models/best.pt")
    else:
        st.warning("No model found. Train in Colab and put best.pt in models/")
    st.divider()
    st.caption(f"Database in use: {config.DB_PATH}")
    st.caption(f"Stored now: {len(db.segments_df())} road segments, {len(db.detections_df())} potholes, "
               f"{len(db.scans_df())} scans")
    if st.button("Load demo data", width="stretch"):
        demo.load()
        st.rerun()
    if st.button("Remove demo data only", width="stretch"):
        db.clear_demo()
        st.rerun()
    confirm = st.checkbox("I want to delete ALL saved data")
    if st.button("Clear all data", width="stretch", disabled=not confirm):
        db.clear()
        st.rerun()


# ---------------------------------------------------------------- dashboard
def build_map(seg, det, show_heat, sev_filter):
    lats = [p[0] for c in seg.coords for p in c]
    lons = [p[1] for c in seg.coords for p in c]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=16, tiles="OpenStreetMap")
    for r in seg.itertuples():
        h = "n/a" if pd.isna(r.health) else f"{r.health:.0f}"
        folium.PolyLine(
            r.coords, color=CAT_COLORS[score.label(r.health)], weight=6, opacity=0.85,
            tooltip=f"{r.name} | health {h} | potholes {r.n_potholes}",
        ).add_to(m)
    d = det[det.severity.isin(sev_filter)]
    if show_heat and not d.empty:
        HeatMap(d[["lat", "lon"]].values.tolist(), radius=18).add_to(m)
    else:
        for r in d.itertuples():
            folium.CircleMarker(
                [r.lat, r.lon], radius=4, color=SEV_COLORS.get(r.severity, "#888"), fill=True, fill_opacity=0.9,
                tooltip=f"{r.severity} | conf {r.confidence:.2f}",
            ).add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return m


@st.fragment
def dashboard():
    seg, det, scans = db.segments_df(), db.detections_df(), db.scans_df()
    if seg.empty:
        st.info("No saved data yet. Run a scan in the Scan area tab, analyse a video, or load demo data.")
        return

    options = ["All saved data"] + [f"#{r.id} {r.name} ({str(r.created_at)[:16]})" for r in scans.itertuples()]
    if st.session_state.get("dash_area") not in options:
        st.session_state["dash_area"] = options[1] if len(options) > 1 else options[0]   # latest scan
    choice = st.selectbox("Show data from", options, key="dash_area")
    if choice != "All saved data":
        sc = scans[scans.id == int(choice.split()[0].lstrip("#"))].iloc[0]
        s0, w0, n0, e0 = sc.south, sc.west, sc.north, sc.east
        seg = seg[seg.coords.apply(lambda c: any(s0 <= p[0] <= n0 and w0 <= p[1] <= e0 for p in c))].copy()
        det = det[det.lat.between(s0, n0) & det.lon.between(w0, e0)]
        if seg.empty:
            st.info("No roads saved for this scan.")
            return

    seg = seg.copy()
    seg["category"] = seg.health.apply(score.label)
    scanned = seg[seg.category != "Not scanned"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Road segments", len(seg))
    k2.metric("Scanned", len(scanned))
    k3.metric("Potholes found", len(det))
    k4.metric("Average health", "n/a" if scanned.empty else f"{scanned.health.mean():.0f} / 100")
    k5.metric("Poor segments", "n/a" if scanned.empty else f"{(scanned.category == 'Poor').mean() * 100:.0f}%")

    f1, f2 = st.columns([3, 1])
    sev_filter = f1.multiselect("Severity shown on map", ["low", "medium", "high"],
                                default=["low", "medium", "high"], key="dash_sev")
    show_heat = f2.toggle("Heatmap", value=False, key="dash_heat")

    left, right = st.columns([3, 2])
    with left:
        map_key = "dash_map_" + str(hash((choice, len(seg), len(det), round(float(seg.health.fillna(-1).sum()), 2),
                                          show_heat, tuple(sev_filter))))
        st_folium(build_map(seg, det, show_heat, sev_filter), height=520, use_container_width=True,
                  returned_objects=[], key=map_key)
        st.caption("Road colour: green good, amber fair, red poor, grey not scanned.")
    with right:
        cat = seg.category.value_counts().reset_index()
        cat.columns = ["category", "segments"]
        fig = px.pie(cat, names="category", values="segments", hole=0.55, color="category",
                     color_discrete_map=CAT_COLORS, title="Road condition")
        fig.update_layout(height=250, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, width="stretch")

        sev = det.severity.value_counts().reindex(["low", "medium", "high"]).fillna(0).reset_index()
        sev.columns = ["severity", "potholes"]
        fig = px.bar(sev, x="severity", y="potholes", color="severity", color_discrete_map=SEV_COLORS,
                     title="Potholes by severity")
        fig.update_layout(height=250, margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        worst = scanned.nsmallest(10, "health").copy()
        worst["road"] = worst["name"] + " #" + worst["id"].astype(str)
        fig = px.bar(worst, x="health", y="road", orientation="h", title="10 worst segments (lower is worse)",
                     range_x=[0, 100])
        fig.update_layout(height=340, yaxis=dict(autorange="reversed"), margin=dict(t=40, b=0))
        fig.update_traces(marker_color="#d63b3b")
        st.plotly_chart(fig, width="stretch")
    with c2:
        t = pd.to_datetime(det.captured_at, errors="coerce", utc=True, format="mixed").dropna()
        if t.empty:
            st.info("No image dates available for a timeline.")
        else:
            tl = t.dt.tz_convert(None).dt.to_period("M").astype(str).value_counts().sort_index().reset_index()
            tl.columns = ["month", "potholes"]
            fig = px.line(tl, x="month", y="potholes", markers=True, title="Detections by image date")
            fig.update_layout(height=340, margin=dict(t=40, b=0))
            st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        src = det.source.value_counts().reset_index()
        src.columns = ["source", "potholes"]
        fig = px.bar(src, x="source", y="potholes", title="Potholes by image source")
        fig.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    with c4:
        fig = px.histogram(scanned, x="health", nbins=10, title="Health score distribution", range_x=[0, 100])
        fig.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Segment table")
    table = seg[["id", "name", "length_m", "images_checked", "n_potholes", "health", "category"]].sort_values("health")
    st.dataframe(table.round(1), width="stretch", hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False), "road_health.csv", "text/csv", key="dash_csv")

    with st.expander("Scan history"):
        if scans.empty:
            st.caption("No area scans saved yet.")
        else:
            st.dataframe(scans[["id", "name", "source", "created_at", "roads", "photos", "potholes"]],
                         width="stretch", hide_index=True)


# --------------------------------------------------------------- test image
def test_tab():
    st.write("Try the model on your own photos or dashcam frames.")
    files = st.file_uploader("Road images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude (optional, to save results)", value=0.0, format="%.6f")
    lon = c2.number_input("Longitude", value=0.0, format="%.6f")

    if st.button("Detect potholes", disabled=not files or not config.MODEL_PATH.exists()):
        results = []
        for f in files:
            dets, annotated = detect.run(f.getvalue(), conf)
            results.append({"name": f.name, "dets": dets, "img": annotated})
        st.session_state["test_results"] = results

    results = st.session_state.get("test_results", [])
    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            st.image(r["img"], caption=f"{r['name']}: {len(r['dets'])} pothole(s)")
    if results and st.button("Save detections to database"):
        if lat == 0 and lon == 0:
            st.error("Enter the latitude and longitude first.")
        else:
            sid = db.nearest_segment(lat, lon)
            db.bump_checked(sid)
            now = datetime.now(timezone.utc).isoformat()
            for r in results:
                for dt in r["dets"]:
                    db.add_detection({"segment_id": sid, "lat": lat, "lon": lon, "confidence": dt["confidence"],
                                      "area_ratio": dt["area_ratio"], "severity": dt["severity"],
                                      "source": "upload", "captured_at": now})
            score.recompute()
            st.success("Saved." if sid else "Saved (no road segment within 40 m, so no health update).")


tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Scan area", "Test images", "Video"])
with tab1:
    dashboard()
with tab2:
    autoscan.render(conf)
with tab3:
    test_tab()
with tab4:
    video_tab.render(conf)