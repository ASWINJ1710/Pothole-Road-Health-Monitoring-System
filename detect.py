import io

from PIL import Image

import config
from score import severity

_model = None


def load():
    global _model
    if _model is None:
        from ultralytics import YOLO  # lazy import so the dashboard works without it
        _model = YOLO(str(config.MODEL_PATH))
    return _model


def run(image_bytes, conf=config.CONF_THRES):
    """Returns (list of detections, annotated PIL image)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    res = load().predict(img, conf=conf, classes=config.CLASS_FILTER, verbose=False)[0]
    w, h = img.size
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
        dets.append({"confidence": float(b.conf[0]), "area_ratio": ratio,
                     "severity": severity(ratio), "box": [x1, y1, x2, y2]})
    annotated = Image.fromarray(res.plot()[..., ::-1])
    return dets, annotated
