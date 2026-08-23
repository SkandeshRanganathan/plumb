import pandas as pd
import kagglehub
import os
import difflib

class PlayerDatabase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PlayerDatabase, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    def _load_data(self):
        try:
            print("Loading Kaggle Player Dataset into memory...")
            # Download/get cached path
            dataset_path = kagglehub.dataset_download("dhavalrupapara/cricket-players-worldwide-dataset")
            csv_file = os.path.join(dataset_path, "players_data_with_all_info.csv")
            
            # Load into pandas dataframe
            self.df = pd.read_csv(csv_file)
            
            # Create a clean list of names for fuzzy matching
            self.df['fullname'] = self.df['fullname'].astype(str)
            self.player_names = self.df['fullname'].tolist()
            
            print(f"Successfully loaded {len(self.player_names)} players into database.")
        except Exception as e:
            print(f"Failed to load Kaggle dataset: {e}")
            self.df = None
            self.player_names = []

    def get_player_profile(self, name: str) -> dict:
        """
        Fuzzy matches the given name against the Kaggle dataset.
        Returns a dict containing battingstyle and bowlingstyle.
        """
        if self.df is None or not name:
            return {"battingstyle": "unknown", "bowlingstyle": "unknown"}

        # Attempt exact match first (case-insensitive)
        exact_match = self.df[self.df['fullname'].str.lower() == name.lower()]
        
        if not exact_match.empty:
            row = exact_match.iloc[0]
            return {
                "fullname": str(row['fullname']),
                "battingstyle": str(row['battingstyle']) if pd.notna(row['battingstyle']) else "unknown",
                "bowlingstyle": str(row['bowlingstyle']) if pd.notna(row['bowlingstyle']) else "unknown"
            }
            
        # Fallback to fuzzy match
        matches = difflib.get_close_matches(name, self.player_names, n=1, cutoff=0.6)
        
        if matches:
            best_match = matches[0]
            row = self.df[self.df['fullname'] == best_match].iloc[0]
            return {
                "fullname": str(row['fullname']),
                "battingstyle": str(row['battingstyle']) if pd.notna(row['battingstyle']) else "unknown",
                "bowlingstyle": str(row['bowlingstyle']) if pd.notna(row['bowlingstyle']) else "unknown"
            }
            
        return {"battingstyle": "unknown", "bowlingstyle": "unknown"}

# Initialize the singleton instance at module load so it doesn't lag during API calls
db_instance = PlayerDatabase()

def get_player_profile(name: str) -> dict:
    return db_instance.get_player_profile(name)
