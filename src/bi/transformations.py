import pandas as pd
from pathlib import Path

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MASTER_DATA_PATH = ROOT_DIR / "data" / "master" / "master_dataset.parquet"
ANOMALY_DATA_PATH = ROOT_DIR / "experiments" / "results" / "anomaly_detections.csv"
BI_OUT_DIR = ROOT_DIR / "data" / "bi"

def run_bi_transformations():
    print("Loading master dataset...")
    df = pd.read_parquet(MASTER_DATA_PATH)
    
    # ---------------------------------------------------------
    # 1. CREATE DIMENSIONS
    # ---------------------------------------------------------
    print("Building Dimensions...")
    
    # dim_bowler
    dim_bowler = df[['bowler_id', 'bowler', 'bowling_style', 'right_armed_bowl']].drop_duplicates(subset=['bowler_id']).copy()
    dim_bowler.to_parquet(BI_OUT_DIR / "dim_bowler.parquet", index=False)
    
    # dim_match
    dim_match = df[['match_id', 'venue', 'city', 'country', 'match_date', 'format']].drop_duplicates(subset=['match_id']).copy()
    # Add a simple surrogate key for weather since weather changes per match
    dim_match['weather_id'] = dim_match['match_id'] 
    dim_match.to_parquet(BI_OUT_DIR / "dim_match.parquet", index=False)
    
    # dim_weather
    dim_weather = df[['match_id', 'temperature_c', 'humidity_pct', 'wind_speed_kmh', 'cloud_cover_pct', 'pitch_type']].drop_duplicates(subset=['match_id']).copy()
    dim_weather.rename(columns={'match_id': 'weather_id'}, inplace=True)
    dim_weather.to_parquet(BI_OUT_DIR / "dim_weather.parquet", index=False)

    # dim_batter
    dim_batter = df[['batter_id', 'batter', 'right_handed_bat']].drop_duplicates(subset=['batter_id']).copy()
    
    # 1. Integrate Wikipedia Centuries Dataset
    wiki_path = ROOT_DIR / "data" / "raw" / "wiki_centuries.csv"
    dim_batter['career_100s'] = 0
    if wiki_path.exists():
        df_wiki = pd.read_csv(wiki_path)
        wiki_dict = df_wiki.set_index('Player')['Total'].to_dict()
        
        for idx, row in dim_batter.iterrows():
            b_name = str(row['batter']).strip()
            
            # Exact match
            if b_name in wiki_dict:
                dim_batter.at[idx, 'career_100s'] = wiki_dict[b_name]
                continue
                
            # Fuzzy/Partial match (e.g. "V Kohli" matches "Virat Kohli")
            parts = b_name.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                first_init = parts[0][0]
                
                for k_name, total_100s in wiki_dict.items():
                    k_parts = k_name.split()
                    if len(k_parts) >= 2:
                        if k_parts[-1] == last_name and k_parts[0][0] == first_init:
                            dim_batter.at[idx, 'career_100s'] = total_100s
                            break
                
    # 2. Integrate Howstat PDF Extracted Data
    howstat_path = ROOT_DIR / "data" / "raw" / "howstat_dismissals.csv"
    if howstat_path.exists():
        df_howstat = pd.read_csv(howstat_path)
        dim_batter = pd.merge(dim_batter, df_howstat, left_on='batter', right_on='Player', how='left')
        dim_batter.drop(columns=['Player'], inplace=True)
    else:
        dim_batter['dismissed_bowled_pct'] = 0.0
        dim_batter['dismissed_lbw_pct'] = 0.0
        dim_batter['dismissed_caught_behind_pct'] = 0.0
        
    dim_batter.to_parquet(BI_OUT_DIR / "dim_batter.parquet", index=False)

    # ---------------------------------------------------------
    # 2. CREATE FACTS
    # ---------------------------------------------------------
    print("Building Facts...")
    
    # fact_deliveries (Grain: 1 row = 1 delivery)
    fact_cols = [
        'delivery_id', 'match_id', 'bowler_id', 'batter_id', 
        'over_num', 'ball_in_over', 'ball_speed_kmh', 'pitch_x', 'pitch_y', 
        'stumps_x', 'stumps_y', 'lateral_swing', 'runs', 'extras', 'is_wide', 'is_no_ball',
        'dismissal_details', 'batter_runs', 'field_x', 'field_y', 'delivery_type'
    ]
    # Only keep columns that actually exist in the dataframe to prevent errors
    valid_fact_cols = [c for c in fact_cols if c in df.columns]
    
    fact_deliveries = df[valid_fact_cols].copy()
    fact_deliveries.to_parquet(BI_OUT_DIR / "fact_deliveries.parquet", index=False)
    
    # fact_anomalies (Grain: 1 row = 1 anomalous delivery)
    if ANOMALY_DATA_PATH.exists():
        print("Building fact_anomalies...")
        df_anomalies = pd.read_csv(ANOMALY_DATA_PATH)
        df_anomalies.to_parquet(BI_OUT_DIR / "fact_anomalies.parquet", index=False)

    print(f"BI Data Mart successfully generated at {BI_OUT_DIR}")

if __name__ == "__main__":
    BI_OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_bi_transformations()