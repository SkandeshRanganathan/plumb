"""
weather_fetch.py  –  MODULE 1-E
Fetches historical weather from Open-Meteo archive API for all
matches that have venue + date information after the CricSheet join.

Open-Meteo archive API (free, no auth):
  https://archive-api.open-meteo.com/v1/archive
  ?latitude=LAT&longitude=LON
  &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
  &hourly=temperature_2m,relativehumidity_2m,windspeed_10m,
          winddirection_10m,precipitation,surface_pressure,cloudcover

Match start times are approximate (cricket typically starts 10:00–11:00 local).
We take the hourly reading closest to 10:30 local time.
"""

import sys
import time
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_WEATHER, DATA_PROCESSED, OPEN_METEO_URL

DATA_WEATHER.mkdir(parents=True, exist_ok=True)

# ── Venue → (latitude, longitude) ───────────────────────────────────────────
# Key international cricket venues
VENUE_COORDS: Dict[str, Tuple[float, float]] = {
    # India
    "Eden Gardens":                         (22.5645, 88.3432),
    "Wankhede Stadium":                     (18.9388, 72.8258),
    "M. Chinnaswamy Stadium":               (12.9794, 77.5996),
    "MA Chidambaram Stadium":               (13.0627, 80.2793),
    "Narendra Modi Stadium":                (23.0928, 72.5978),
    "Arun Jaitley Stadium":                 (28.6345, 77.2198),
    "Rajiv Gandhi International Stadium":   (17.4062, 78.5429),
    "Punjab Cricket Association Stadium":   (30.6947, 76.7812),
    "Sawai Mansingh Stadium":               (26.9124, 75.8234),
    "Himachal Pradesh Cricket Association": (32.2221, 76.3148),
    "JSCA International Stadium Complex":   (23.3441, 85.3096),
    "Vidarbha Cricket Association Stadium": (21.1458, 79.0882),
    "Dr DY Patil Sports Academy":           (19.0596, 73.0163),
    "Brabourne Stadium":                    (18.9395, 72.8264),
    # England
    "Lord's Cricket Ground":                (51.5297, -0.1727),
    "The Oval":                             (51.4836, -0.1149),
    "Edgbaston":                            (52.4556, -1.9025),
    "Old Trafford":                         (53.4567, -2.2873),
    "Headingley":                           (53.8173, -1.5823),
    "Trent Bridge":                         (52.9368,  -1.1322),
    "The Ageas Bowl":                       (50.9253, -1.3253),
    "County Ground, Bristol":               (51.4555, -2.6117),
    # Australia
    "Melbourne Cricket Ground":             (-37.8200, 144.9834),
    "Sydney Cricket Ground":                (-33.8915, 151.2245),
    "Gabba":                                (-27.4858, 153.0381),
    "Adelaide Oval":                        (-34.9154, 138.5961),
    "WACA Ground":                          (-31.9593, 115.8628),
    "Optus Stadium":                        (-31.9512, 115.8862),
    "Bellerive Oval":                       (-42.8806, 147.3456),
    "Manuka Oval":                          (-35.3184, 149.1416),
    # South Africa
    "Newlands Cricket Ground":              (-33.9259,  18.4340),
    "Wanderers Stadium":                    (-26.1436,  28.0603),
    "Kingsmead":                            (-29.8495,  31.0226),
    "St George's Park":                     (-33.9558,  25.6010),
    "SuperSport Park":                      (-25.7528,  28.2282),
    # New Zealand
    "Basin Reserve":                        (-41.3192, 174.7782),
    "Eden Park":                            (-36.8752, 174.7439),
    "Hagley Oval":                          (-43.5320, 172.6270),
    "Seddon Park":                          (-37.7881, 175.2834),
    "McLean Park":                          (-39.4864, 176.8989),
    # Pakistan
    "National Stadium Karachi":             (24.8774,  67.0618),
    "Gaddafi Stadium":                      (31.5144,  74.3396),
    "Rawalpindi Cricket Stadium":           (33.6092,  73.0679),
    # Sri Lanka
    "P Sara Oval":                          ( 6.9101,  79.8684),
    "Sinhalese Sports Club":                ( 6.9023,  79.8736),
    "Galle International Stadium":          ( 6.0280,  80.2112),
    "Pallekele International Stadium":      ( 7.2867,  80.6267),
    # West Indies
    "Kensington Oval":                      (13.0813, -59.6144),
    "Sabina Park":                          (17.9973, -76.7789),
    "Queen's Park Oval":                    (10.6500, -61.4167),
    "Bourda Oval":                          ( 6.8050, -58.1579),
    # Bangladesh
    "Shere Bangla National Stadium":        (23.7804,  90.3592),
    "Zahur Ahmed Chowdhury Stadium":        (22.3569,  91.8378),
    # UAE
    "Dubai International Cricket Stadium": (25.0421,  55.2239),
    "Sheikh Zayed Stadium":                (24.4616,  54.3657),
    "Sharjah Cricket Stadium":             (25.3368,  55.3730),
    # Zimbabwe
    "Harare Sports Club":                  (-17.8165,  31.0378),
    "Queens Sports Club":                  (-20.1536,  28.5827),
}


def get_venue_coords(venue: Optional[str]) -> Optional[Tuple[float, float]]:
    """Try to find lat/lon for a venue string (partial matching)."""
    if not venue:
        return None
    # Exact match first
    if venue in VENUE_COORDS:
        return VENUE_COORDS[venue]
    # Partial match
    venue_lower = venue.lower()
    for vname, coords in VENUE_COORDS.items():
        if vname.lower() in venue_lower or venue_lower in vname.lower():
            return coords
    return None


def fetch_weather_for_date(
    lat: float, lon: float, date_str: str, cache_dir: Path
) -> Optional[Dict]:
    """
    Fetch Open-Meteo hourly data for one location+date.
    Caches results to avoid repeated API calls.
    Returns dict of weather variables at hour 10 (10:00 local), or None.
    """
    cache_key = f"{lat:.4f}_{lon:.4f}_{date_str}"
    cache_file = cache_dir / f"{cache_key}.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date":   date_str,
        "hourly": ",".join([
            "temperature_2m",
            "relativehumidity_2m",
            "windspeed_10m",
            "winddirection_10m",
            "precipitation",
            "surface_pressure",
            "cloudcover",
        ]),
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return None

    # Extract hour 10 reading (10:00 local ≈ typical match start)
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    target_suffix = "T10:00"
    idx = None
    for i, t in enumerate(times):
        if t.endswith(target_suffix):
            idx = i
            break

    if idx is None and times:
        idx = 10  # fallback to 10th hour

    if idx is None or not hourly:
        return None

    result = {
        "temperature_c":    hourly.get("temperature_2m",      [None] * 25)[idx],
        "humidity_pct":     hourly.get("relativehumidity_2m", [None] * 25)[idx],
        "wind_speed_kmh":   hourly.get("windspeed_10m",       [None] * 25)[idx],
        "wind_direction_deg": hourly.get("winddirection_10m", [None] * 25)[idx],
        "precipitation_mm": hourly.get("precipitation",       [None] * 25)[idx],
        "pressure_hpa":     hourly.get("surface_pressure",    [None] * 25)[idx],
        "cloud_cover_pct":  hourly.get("cloudcover",          [None] * 25)[idx],
        "source": "open-meteo",
        "date":   date_str,
        "lat":    lat,
        "lon":    lon,
    }

    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


def run_weather_fetch(hawkeye_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry: iterate over unique (venue, match_date) pairs that have
    HIGH or MEDIUM join confidence, fetch weather, and merge back.
    Returns enriched DataFrame.
    """
    print("=" * 65)
    print("MODULE 1-E: Weather Fetch (Open-Meteo Archive)")
    print("=" * 65)

    cache_dir = DATA_WEATHER / "hourly_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Only fetch weather for MEDIUM or HIGH confidence venue joins
    eligible = hawkeye_df[
        hawkeye_df["venue_join_confidence"].isin(["HIGH", "MEDIUM"]) &
        hawkeye_df["venue"].notna() &
        hawkeye_df["match_date"].notna()
    ].copy()

    if eligible.empty:
        print("  No eligible rows with venue+date. Skipping weather fetch.")
        return hawkeye_df

    # Get unique (venue, date) pairs
    unique_pairs = eligible[["venue", "match_date"]].drop_duplicates()
    print(f"  Unique (venue, date) pairs to fetch: {len(unique_pairs)}")

    weather_rows = []
    fetched, skipped, failed = 0, 0, 0

    for _, row in unique_pairs.iterrows():
        venue = row["venue"]
        date = row["match_date"]
        if pd.isna(date):
            skipped += 1
            continue
        date_str = str(date)[:10]
        coords = get_venue_coords(venue)
        if coords is None:
            skipped += 1
            continue
        lat, lon = coords
        wx = fetch_weather_for_date(lat, lon, date_str, cache_dir)
        if wx is None:
            failed += 1
            continue
        wx["venue"] = venue
        wx["match_date"] = date_str
        weather_rows.append(wx)
        fetched += 1
        time.sleep(0.1)   # be polite to the API

    print(f"  Fetched: {fetched}  |  Skipped (no coords): {skipped}  |  Failed: {failed}")

    if not weather_rows:
        print("  No weather data retrieved.")
        return hawkeye_df

    weather_df = pd.DataFrame(weather_rows)
    weather_df.to_csv(DATA_WEATHER / "match_weather.csv", index=False)
    print(f"  Saved weather data: {DATA_WEATHER / 'match_weather.csv'}")

    # Merge back onto hawkeye_df
    # Normalise match_date to str for joining
    hawkeye_df["_date_str"] = hawkeye_df["match_date"].astype(str).str[:10]
    weather_df["_date_str"] = weather_df["match_date"]

    weather_cols = ["venue", "_date_str", "temperature_c", "humidity_pct",
                    "wind_speed_kmh", "wind_direction_deg", "precipitation_mm",
                    "pressure_hpa", "cloud_cover_pct"]

    hawkeye_df = hawkeye_df.merge(
        weather_df[weather_cols],
        on=["venue", "_date_str"],
        how="left",
        suffixes=("_old", "")
    )
    for col in ["temperature_c", "humidity_pct", "wind_speed_kmh",
                "wind_direction_deg", "precipitation_mm", "pressure_hpa",
                "cloud_cover_pct"]:
        old = col + "_old"
        if old in hawkeye_df.columns:
            hawkeye_df[col] = hawkeye_df[col].fillna(hawkeye_df[old])
            hawkeye_df.drop(columns=[old], inplace=True)

    hawkeye_df["weather_available"] = hawkeye_df["temperature_c"].notna().astype(int)
    hawkeye_df.drop(columns=["_date_str"], inplace=True)

    wx_pct = hawkeye_df["weather_available"].mean() * 100
    print(f"  Weather coverage: {wx_pct:.1f}% of all deliveries")

    return hawkeye_df


if __name__ == "__main__":
    parquet_path = DATA_PROCESSED / "hawkeye_with_venue.parquet"
    if not parquet_path.exists():
        print("Run cricsheet_join.py first.")
        sys.exit(1)
    df = pd.read_parquet(parquet_path)
    df = run_weather_fetch(df)
    df.to_parquet(DATA_PROCESSED / "hawkeye_with_weather.parquet", index=False)
    print(f"\nSaved hawkeye_with_weather.parquet  ({len(df):,} rows)")
