import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent
DB_PATH = BASE / "data" / "potholes.db"
IMG_DIR = BASE / "data" / "detections"
MODEL_PATH = BASE / "models" / "best.pt"

MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

CONF_THRES = 0.35
CLASS_FILTER = None  # e.g. [0] to keep only class id 0 if your dataset has many classes
SEGMENT_LEN_M = 100