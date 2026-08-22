"""
config.py
Project-wide configuration constants and paths.
"""
import os
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_MASTER     = ROOT / "data" / "master"
DATA_EXTERNAL   = ROOT / "data" / "external"
DATA_CRICSHEET  = DATA_EXTERNAL / "cricsheet"
DATA_WEATHER    = DATA_EXTERNAL / "weather"
DATA_BOWLER     = ROOT / "data" / "bowler_profiles"
DATA_BALL_STATE = ROOT / "data" / "ball_state"
EXPERIMENTS     = ROOT / "experiments"
MODELS_SAVED    = ROOT / "models" / "saved"

# ── Raw dataset source paths ──────────────────────────────────────────────────
HAWKEYE_DIR     = ROOT / "datasets" / "hawkeye_stats"
CRIC360_DIR     = ROOT / "datasets" / "cric360"
KAGGLE_CACHE    = Path.home() / ".cache" / "kagglehub" / "datasets"

HAWKEYE_FILES = {
    "IPL_Men":   HAWKEYE_DIR / "mensIPLHawkeyeStats.csv",
    "ODI_Men":   HAWKEYE_DIR / "mensODIHawkeyeStats.csv",
    "Test_Men":  HAWKEYE_DIR / "mensTestHawkeyeStats.csv",
    "IPL_Women": HAWKEYE_DIR / "womensIPLHawkeyeStats.csv",
    "ODI_Women": HAWKEYE_DIR / "womensODIHawkeyeStats.csv",
    "Test_Women":HAWKEYE_DIR / "womensTestHawkeyeStats.csv",
}

# ── Coordinate sanity bounds ─────────────────────────────────────────────────
# Cricket pitch is 20.12m (22 yards) between stumps.
# pitchX = lateral offset from centre (metres); valid: ±5m
# pitchY = distance from bowler's stumps to bounce (metres); valid: 0-25m
# stumpsX = lateral position at batter's stumps (metres); valid: ±5m
# stumpsY = height at stumps (metres); valid: 0-5m
COORD_BOUNDS = {
    "pitchX_min": -5.0, "pitchX_max": 5.0,
    "pitchY_min":  0.01, "pitchY_max": 25.0,
    "stumpsX_min":-5.0, "stumpsX_max": 5.0,
    "stumpsY_min": 0.01, "stumpsY_max": 5.0,
    "speed_min_ms": 5.0,   # ~18 km/h – below this is sensor noise
    "speed_max_ms": 50.0,  # ~180 km/h – above this is sensor error
}

# ── Pitch length constants (metres) ──────────────────────────────────────────
PITCH_LENGTH_M   = 20.12   # stumps to stumps (22 yards)
CREASE_OFFSET_M  = 1.22    # popping crease to stumps

# ── Delivery length classification (pitchY from BOWLER stumps) ───────────────
# The pitchY in HawkeyeStats measures from the BOWLER's stumps to the bounce point.
# So "close to batter" means HIGH pitchY. Standard lengths:
DELIVERY_LENGTHS = {
    # (min_pitchY, max_pitchY): label
    (0.0,  2.0):  "full_toss",        # ball never pitches (or pitches in crease)
    (2.0,  5.5):  "yorker",           # pitches in/near batter's crease (~2-4m from bat)
    (5.5,  8.0):  "full",             # full length (drives encouraged)
    (8.0, 11.0):  "good_length",      # optimal seam/swing length
    (11.0,13.5):  "short_of_length",  # back of a length
    (13.5,25.0):  "short",            # short / bouncer
}

# ── Height classification at stumps ──────────────────────────────────────────
STUMPS_HEIGHT_M = 0.711   # top of bails
STUMPS_HEIGHT_CLASSES = {
    (0.0,   0.30):  "below_knee",
    (0.30,  0.55):  "knee",
    (0.55,  0.80):  "hip",
    (0.80,  1.10):  "waist",
    (1.10,  1.45):  "chest",
    (1.45,  5.0):   "head",
}

# ── Wide detection threshold (stumpsX > threshold relative to stumps) ────────
# T20/ODI: ball passing outside off or leg stump line when batter stands still
# Dynamic wide: adjusted by batter position (needs batter_x, not available in HawkeyeStats)
WIDE_THRESHOLD_X = 0.535  # half-width of stumps (3 stumps × 0.083m + gaps ≈ 0.228m / 2)

# ── Ball age phases (Test cricket) ───────────────────────────────────────────
# In Test cricket new ball available at over 80 of innings
NEW_BALL_OVERS = [0, 80, 160]
BALL_PHASES = {
    (0,  10):  "new_ball",
    (10, 35):  "swinging",
    (35, 55):  "worn",
    (55, 80):  "old",
    (80, 90):  "second_new_ball",
    (90, 160): "second_worn",
}

# ── Ball type by format/country (rule-based) ─────────────────────────────────
BALL_TYPE_RULES = {
    ("Test_Men",  "India"):    "SG",
    ("Test_Men",  "England"):  "Dukes",
    ("Test_Men",  "default"):  "Kookaburra",
    ("Test_Women","default"):  "Kookaburra",
    ("ODI_Men",   "default"):  "Kookaburra_White",
    ("ODI_Women", "default"):  "Kookaburra_White",
    ("IPL_Men",   "default"):  "Kookaburra_Pink",
    ("IPL_Women", "default"):  "Kookaburra_Pink",
}

# ── Venue → country/city mapping (stub; expanded by CricSheet join) ──────────
VENUE_COUNTRY_MAP = {}   # populated at runtime from cricsheet data

# ── Open-Meteo API base URL ───────────────────────────────────────────────────
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# ── CricSheet base URL ────────────────────────────────────────────────────────
CRICSHEET_BASE = "https://cricsheet.org/downloads"

# ── Experiment reproducibility seed ──────────────────────────────────────────
RANDOM_SEED = 42
