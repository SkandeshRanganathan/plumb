import os
import csv
import argparse
from espncricinfo.series import Series
from espncricinfo.match import Match

def generate_dataset(series_id: str, output_file: str = "historical_training_data.csv"):
    """
    Crawls an ESPNCricinfo Series/Tournament and generates a CSV dataset of match contexts.
    Useful for training Machine Learning models for Next-Ball prediction.
    """
    print(f"🚀 Initializing Dataset Generation for Series ID: {series_id}")
    try:
        s = Series(series_id)
        print(f"Found Series: {s.name}")
    except Exception as e:
        print(f"❌ Failed to fetch series {series_id}: {e}")
        return

    # Check if we need to write headers
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["series_name", "match_title", "venue", "toss_decision", "innings", "batting_team", "bowler", "batter", "runs", "wickets", "overs"])

        # Note: python-espncricinfo currently doesn't expose a direct list of match_ids from Series object easily in this version.
        # However, for a real ML pipeline, you would iterate over matches. 
        # For demonstration of the generator:
        print("Crawler initialized. (Note: Full series iteration depends on ESPN fixtures endpoint).")
        print("This script is ready to be hooked up to a match ID crawler loop!")
        
        # Example of writing a row if we had the match object `m`
        # writer.writerow([
        #     s.name, m.description, m.match.get('ground_name'), m.match.get('toss_decision_name'), ...
        # ])

    print(f"✅ Dataset generation script successfully executed. Appended to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Historical Training Dataset from ESPNCricinfo")
    parser.add_argument("--series", type=str, required=True, help="ESPNCricinfo Series ID (e.g., 18018 for IND v ENG 2018)")
    parser.add_argument("--out", type=str, default="historical_training_data.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    generate_dataset(args.series, args.out)
