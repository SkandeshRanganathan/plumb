import pandas as pd
import numpy as np

files = {
    'IPL_Men': 'datasets/hawkeye_stats/mensIPLHawkeyeStats.csv',
    'ODI_Men': 'datasets/hawkeye_stats/mensODIHawkeyeStats.csv',
    'Test_Men': 'datasets/hawkeye_stats/mensTestHawkeyeStats.csv',
    'IPL_Women': 'datasets/hawkeye_stats/womensIPLHawkeyeStats.csv',
    'ODI_Women': 'datasets/hawkeye_stats/womensODIHawkeyeStats.csv',
    'Test_Women': 'datasets/hawkeye_stats/womensTestHawkeyeStats.csv',
}

summary = []
for fmt, fpath in files.items():
    df = pd.read_csv(fpath, low_memory=False)
    for col in ['pitchX','pitchY','stumpsX','stumpsY','ballSpeed','fieldX','fieldY']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    valid = df[
        (df['pitchX'].notna()) &
        (df['pitchX'].abs() < 5) &
        (df['pitchY'].notna()) &
        (df['pitchY'] > 0) &
        (df['pitchY'] < 25) &
        (df['stumpsX'].notna()) &
        (df['stumpsX'].abs() < 5) &
        (df['stumpsY'].notna()) &
        (df['stumpsY'] > 0) &
        (df['stumpsY'] < 5)
    ]
    speed_valid = valid[valid['ballSpeed'] > 0]['ballSpeed']
    wides = df['extras'].str.contains('Wd', na=False).sum()
    noballs = df['extras'].str.contains('Nb', na=False).sum()
    styles = df['bowlingStyle'].value_counts().to_dict()

    row = {
        'format': fmt,
        'total_rows': len(df),
        'valid_trajectory_rows': len(valid),
        'unique_matches': df['matchId'].nunique(),
        'unique_bowlers': df['bowler'].nunique(),
        'unique_batters': df['batter'].nunique(),
        'wides': wides,
        'no_balls': noballs,
        'speed_valid_count': len(speed_valid),
        'speed_min_kmh': round(speed_valid.min() * 3.6, 1) if len(speed_valid) > 0 else None,
        'speed_max_kmh': round(speed_valid.max() * 3.6, 1) if len(speed_valid) > 0 else None,
        'speed_mean_kmh': round(speed_valid.mean() * 3.6, 1) if len(speed_valid) > 0 else None,
        'bowling_styles': str(styles),
        'has_fieldXY': int(df['fieldX'].notna().sum()),
    }
    summary.append(row)
    print("Done:", fmt)

sdf = pd.DataFrame(summary)
print(sdf.to_string())
sdf.to_csv('inspect_output/hawkeye_summary.csv', index=False)
print("Saved to inspect_output/hawkeye_summary.csv")
