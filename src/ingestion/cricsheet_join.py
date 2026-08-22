"""
cricsheet_join.py  –  MODULE 1-D
Downloads CricSheet match-level metadata (YAML ball-by-ball data)
and joins it with the HawkeyeStats data to supply:
  - venue
  - city
  - country
  - match_date
  - teams
  - toss info
  - match result

CricSheet YAML structure (per match file):
  info:
    venue: "Eden Gardens"
    city: "Kolkata"
    dates: ["2023-11-19"]
    teams: ["India", "Australia"]
    match_type: "Test"
    ...

Join strategy:
  1. Load all CricSheet match YAML files matching the relevant format.
  2. Build a lookup: (team_a, team_b, date, format) → {venue, city, country, ...}
  3. For each HawkeyeStats matchId, find the best CricSheet match by
     cross-referencing bowler/batter names appearing in deliveries.
  4. Confidence: HIGH if bowler+batter overlap > 60%, MEDIUM otherwise.
"""

import sys
import re
import json
import zipfile
import requests
import io
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_CRICSHEET, DATA_PROCESSED, CRICSHEET_BASE, RANDOM_SEED

DATA_CRICSHEET.mkdir(parents=True, exist_ok=True)


# ── Country mapping (cricsheet uses country names in venues) ─────────────────
KNOWN_COUNTRIES = {
    "India": "India", "Mumbai": "India", "Delhi": "India",
    "Chennai": "India", "Kolkata": "India", "Hyderabad": "India",
    "Bengaluru": "India", "Ahmedabad": "India", "Pune": "India",
    "Jaipur": "India", "Chandigarh": "India", "Dharamsala": "India",
    "Mohali": "India", "Ranchi": "India", "Nagpur": "India",
    "England": "England", "London": "England", "Birmingham": "England",
    "Manchester": "England", "Leeds": "England", "Nottingham": "England",
    "Southampton": "England", "Bristol": "England",
    "Australia": "Australia", "Melbourne": "Australia",
    "Sydney": "Australia", "Brisbane": "Australia",
    "Adelaide": "Australia", "Perth": "Australia", "Hobart": "Australia",
    "South Africa": "South Africa", "Cape Town": "South Africa",
    "Johannesburg": "South Africa", "Durban": "South Africa",
    "Port Elizabeth": "South Africa", "Pretoria": "South Africa",
    "New Zealand": "New Zealand", "Wellington": "New Zealand",
    "Auckland": "New Zealand", "Christchurch": "New Zealand",
    "Hamilton": "New Zealand", "Napier": "New Zealand",
    "Pakistan": "Pakistan", "Karachi": "Pakistan",
    "Lahore": "Pakistan", "Rawalpindi": "Pakistan",
    "Sri Lanka": "Sri Lanka", "Colombo": "Sri Lanka",
    "Galle": "Sri Lanka", "Kandy": "Sri Lanka",
    "West Indies": "West Indies", "Bridgetown": "West Indies",
    "Georgetown": "West Indies", "Kingston": "West Indies",
    "Port of Spain": "West Indies", "Antigua": "West Indies",
    "Bangladesh": "Bangladesh", "Dhaka": "Bangladesh",
    "Chittagong": "Bangladesh", "Mirpur": "Bangladesh",
    "Zimbabwe": "Zimbabwe", "Harare": "Zimbabwe", "Bulawayo": "Zimbabwe",
    "Afghanistan": "Afghanistan", "Kabul": "Afghanistan",
    "Ireland": "Ireland", "Dublin": "Ireland",
    "UAE": "UAE", "Dubai": "UAE", "Abu Dhabi": "UAE", "Sharjah": "UAE",
}


def city_to_country(city: Optional[str], venue: Optional[str] = None) -> Optional[str]:
    """Map city or venue string to country."""
    for src in [city, venue]:
        if not src:
            continue
        for key, country in KNOWN_COUNTRIES.items():
            if key.lower() in src.lower():
                return country
    return None


def download_cricsheet_zip(format_code: str, dest_dir: Path) -> Optional[Path]:
    """
    Download a CricSheet zip file for the given format.
    format_code: 'tests', 'odis', 't20s', 'ipl' (CricSheet conventions)
    Returns path to extracted directory, or None on failure.
    """
    url = f"{CRICSHEET_BASE}/{format_code}_json.zip"
    local_zip = dest_dir / f"{format_code}_json.zip"
    local_dir = dest_dir / f"{format_code}_json"

    if local_dir.exists() and any(local_dir.glob("*.json")):
        print(f"  [cache] CricSheet {format_code} already extracted at {local_dir}")
        return local_dir

    print(f"  Downloading CricSheet {format_code} from {url} ...")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(local_zip, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(local_dir)
        local_zip.unlink()
        print(f"  ✓ Extracted {len(list(local_dir.glob('*.json')))} match files")
        return local_dir
    except Exception as e:
        print(f"  [WARN] Could not download CricSheet {format_code}: {e}")
        return None


def build_cricsheet_lookup(json_dir: Path) -> pd.DataFrame:
    """
    Parse all CricSheet JSON files into a match metadata DataFrame.
    CricSheet JSON format (v1.0.0):
      {
        "meta": {...},
        "info": {
          "venue": "...", "city": "...", "dates": [...],
          "teams": [...], "match_type": "...", "toss": {...}, ...
        },
        "innings": [...]
      }
    Returns DataFrame with columns:
      cs_match_id, venue, city, country, match_date, format,
      team1, team2, players (list), bowlers (list)
    """
    rows = []
    files = list(json_dir.glob("*.json"))
    print(f"  Parsing {len(files)} CricSheet JSON files ...")

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            info = data.get("info", {})
            dates = info.get("dates", [])
            teams = info.get("teams", [])
            venue = info.get("venue", None)
            city = info.get("city", None)
            country = city_to_country(city, venue)

            # Collect all player names from innings
            bowlers, batters = set(), set()
            for inning in data.get("innings", []):
                for over_data in inning.get("overs", []):
                    for delivery in over_data.get("deliveries", []):
                        bowlers.add(delivery.get("bowler", ""))
                        batters.add(delivery.get("batter", ""))

            rows.append({
                "cs_match_id":   fp.stem,
                "venue":         venue,
                "city":          city,
                "country":       country,
                "match_date":    dates[0] if dates else None,
                "cs_format":     info.get("match_type", ""),
                "team1":         teams[0] if len(teams) > 0 else None,
                "team2":         teams[1] if len(teams) > 1 else None,
                "bowlers":       list(bowlers - {""}),
                "batters":       list(batters - {""}),
                "toss_winner":   info.get("toss", {}).get("winner"),
                "toss_decision": info.get("toss", {}).get("decision"),
                "outcome":       str(info.get("outcome", {})),
            })
        except Exception as e:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    print(f"  ✓ Parsed {len(df)} CricSheet matches")
    return df


def join_hawkeye_to_cricsheet(
    hawkeye_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    format_: str
) -> pd.DataFrame:
    """
    For each unique matchId in hawkeye_df, find the best CricSheet match
    by player overlap (bowlers + batters appearing in that HawkEye match).

    Returns a DataFrame: match_id → {venue, city, country, match_date,
                                     venue_join_confidence, cs_match_id}
    """
    print(f"\n  Joining {format_} HawkEye matchIds to CricSheet ...")

    # Build set of bowler+batter names per hawkeye matchId
    hk_match_players: Dict[int, set] = {}
    for mid, grp in hawkeye_df.groupby("match_id"):
        players = set(grp["bowler"].dropna().unique()) | set(grp["batter"].dropna().unique())
        hk_match_players[mid] = players

    results = []
    for match_id, hk_players in hk_match_players.items():
        best_cs_id    = None
        best_overlap  = 0.0
        best_conf     = "NULL"
        best_venue    = None
        best_city     = None
        best_country  = None
        best_date     = None

        if cs_df.empty:
            results.append({
                "match_id": match_id,
                "venue": None, "city": None, "country": None,
                "match_date": None,
                "venue_join_confidence": "NULL",
                "cs_match_id": None,
            })
            continue

        for _, cs_row in cs_df.iterrows():
            cs_players = set(cs_row.get("bowlers", [])) | set(cs_row.get("batters", []))
            if not cs_players:
                continue
            overlap = len(hk_players & cs_players) / max(len(hk_players), 1)
            if overlap > best_overlap:
                best_overlap  = overlap
                best_cs_id    = cs_row["cs_match_id"]
                best_venue    = cs_row["venue"]
                best_city     = cs_row["city"]
                best_country  = cs_row["country"]
                best_date     = cs_row["match_date"]

        if best_overlap >= 0.6:
            best_conf = "HIGH"
        elif best_overlap >= 0.30:
            best_conf = "MEDIUM"
        elif best_overlap > 0:
            best_conf = "LOW"
        else:
            best_conf = "NULL"

        results.append({
            "match_id":              match_id,
            "venue":                 best_venue,
            "city":                  best_city,
            "country":               best_country,
            "match_date":            best_date,
            "venue_join_confidence": best_conf,
            "cs_match_id":           best_cs_id,
            "player_overlap_score":  round(best_overlap, 3),
        })

    result_df = pd.DataFrame(results)
    conf_counts = result_df["venue_join_confidence"].value_counts()
    print(f"  Join confidence distribution:\n{conf_counts.to_string()}")
    return result_df


def run_cricsheet_join(hawkeye_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry: downloads CricSheet data for all formats, builds match
    metadata lookup, and returns enriched hawkeye_df with venue/date/country.
    """
    print("=" * 65)
    print("MODULE 1-D: CricSheet Venue/Date Join")
    print("=" * 65)

    # Map HawkEye format names to CricSheet download codes
    format_map = {
        "Test_Men":  "tests",
        "ODI_Men":   "odis",
        "IPL_Men":   "ipl",
        "Test_Women":"tests",   # CricSheet has separate women's sets but start with mens
        "ODI_Women": "odis",
        "IPL_Women": "ipl",
    }

    all_match_meta = pd.DataFrame()

    for cs_code in set(format_map.values()):
        json_dir = download_cricsheet_zip(cs_code, DATA_CRICSHEET)
        if json_dir is None:
            print(f"  [SKIP] CricSheet {cs_code} unavailable")
            continue
        cs_df = build_cricsheet_lookup(json_dir)
        if not cs_df.empty:
            all_match_meta = pd.concat([all_match_meta, cs_df], ignore_index=True)

    if all_match_meta.empty:
        print("  [WARN] No CricSheet data available. Venue/date will remain NULL.")
        return hawkeye_df

    all_match_meta.to_parquet(DATA_CRICSHEET / "cricsheet_metadata.parquet", index=False)
    print(f"\n  CricSheet total matches parsed: {len(all_match_meta)}")

    # Join each format
    enriched_parts = []
    for format_, grp in hawkeye_df.groupby("format"):
        join_result = join_hawkeye_to_cricsheet(grp, all_match_meta, format_)
        # Merge back onto the group
        merged = grp.merge(
            join_result[["match_id", "venue", "city", "country",
                          "match_date", "venue_join_confidence",
                          "cs_match_id", "player_overlap_score"]],
            on="match_id", how="left", suffixes=("_old", "")
        )
        # Prefer newly joined values over old placeholders
        for col in ["venue", "city", "country", "match_date",
                    "venue_join_confidence"]:
            old_col = col + "_old"
            if old_col in merged.columns:
                merged[col] = merged[col].fillna(merged[old_col])
                merged.drop(columns=[old_col], inplace=True)
        enriched_parts.append(merged)

    enriched = pd.concat(enriched_parts, ignore_index=True)

    # Infer country from city where still missing
    mask = enriched["country"].isna() & enriched["city"].notna()
    enriched.loc[mask, "country"] = enriched.loc[mask, "city"].apply(
        lambda c: city_to_country(c)
    )

    high = (enriched["venue_join_confidence"] == "HIGH").sum()
    med  = (enriched["venue_join_confidence"] == "MEDIUM").sum()
    low  = (enriched["venue_join_confidence"] == "LOW").sum()
    total = len(enriched)
    print(f"\n  Final join results ({total:,} deliveries):")
    print(f"    HIGH confidence: {high:,} ({100*high/total:.1f}%)")
    print(f"    MEDIUM:          {med:,}  ({100*med/total:.1f}%)")
    print(f"    LOW:             {low:,}  ({100*low/total:.1f}%)")
    print(f"    NULL:            {total-high-med-low:,}")

    return enriched


if __name__ == "__main__":
    # Quick test: load cleaned parquet and run join
    parquet_path = DATA_PROCESSED / "hawkeye_clean.parquet"
    if not parquet_path.exists():
        print("Run hawkeye_ingest.py first to generate hawkeye_clean.parquet")
        sys.exit(1)
    df = pd.read_parquet(parquet_path)
    enriched = run_cricsheet_join(df)
    enriched.to_parquet(DATA_PROCESSED / "hawkeye_with_venue.parquet", index=False)
    print(f"\nSaved: hawkeye_with_venue.parquet  ({len(enriched):,} rows)")
