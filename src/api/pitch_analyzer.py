import json

class PitchIntelligenceEngine:
    def __init__(self):
        # Base scores
        self.base_par_score_t20 = 165
        self.base_par_score_odi = 280
        self.base_par_score_test = 300  # Lowered base Test score for realism
        
    def analyze(self, pitch_type: str, pitch_report_text: str, match_format: str = "T20"):
        """
        Analyzes the Pitch conditions (from CV and Text) to generate a Game Theory Matrix.
        """
        pitch_type = pitch_type.lower()
        pitch_report = pitch_report_text.lower()
        
        # 1. Advanced NLP Keyword Matching
        is_green = any(w in pitch_type for w in ["green", "seam"]) or any(w in pitch_report for w in ["grass", "swing", "seam movement", "overcast", "cloud"])
        is_dry = any(w in pitch_type for w in ["dry", "dust"]) or any(w in pitch_report for w in ["spin", "cracks", "dry", "dust", "turn", "wear", "slow"])
        is_damp = any(w in pitch_type for w in ["damp"]) or any(w in pitch_report for w in ["sticky", "moisture", "rain", "wet"])
        is_bouncy = any(w in pitch_report for w in ["bounce", "pace friendly", "hard", "carry"])
        is_belter = any(w in pitch_report for w in ["flat", "batting paradise", "belter", "nothing for the bowlers", "true", "run feast", "fireworks", "high scoring", "comes onto the bat"])
        
        # 2. Base Adjustments
        if match_format == "T20":
            par_score = self.base_par_score_t20
        elif match_format == "ODI":
            par_score = self.base_par_score_odi
        else:
            par_score = self.base_par_score_test
            
        toss_decision = "Bat First"
        win_prob_bat_1st = 50.0
        
        wear_heatmap_type = "standard"
        optimal_length = "Good Length (6m-8m)"
        
        # 3. Dynamic Additive Scoring Adjustments
        score_modifier = 0
        
        if is_damp:
            score_modifier -= (35 if match_format == "T20" else 80)
            toss_decision = "Bowl First"
            win_prob_bat_1st -= 25.0
            wear_heatmap_type = "damp_sticky"
            optimal_length = "Back of a Length (8m-10m) to let the ball misbehave off the sticky surface."
            
        if is_green:
            score_modifier -= (15 if match_format == "T20" else 40)
            toss_decision = "Bowl First" if not is_damp else toss_decision
            win_prob_bat_1st -= 15.0
            wear_heatmap_type = "green_seaming" if wear_heatmap_type == "standard" else wear_heatmap_type
            if optimal_length == "Good Length (6m-8m)":
                optimal_length = "Full (4m-6m) to invite the drive and induce edges with the movement."
                
        if is_bouncy:
            score_modifier -= (10 if match_format == "T20" else 20)
            win_prob_bat_1st -= 5.0
            
        if is_dry:
            # Spin friendly, gets worse as match progresses
            score_modifier -= (10 if match_format == "T20" else 30)
            toss_decision = "Bat First"  # Bat before it deteriorates
            win_prob_bat_1st += 20.0
            wear_heatmap_type = "dusty_spinning"
            optimal_length = "Hard Length (8m) to extract uneven bounce and let the cracks do the work."
            
        if is_belter:
            # Run feast! Overrides negative score modifiers heavily in white-ball cricket.
            # E.g. Grass + Belter in T20 = Ball comes onto the bat nicely!
            score_modifier += (35 if match_format == "T20" else 100)
            toss_decision = "Bowl First" if match_format == "T20" else "Bat First" # T20 chasing is easier on a belter
            win_prob_bat_1st += (5.0 if match_format == "T20" else 15.0)
            wear_heatmap_type = "flat_belter"
            optimal_length = "Yorkers and Wide Lines. Pitch is flat, rely on defensive variations and pace off."

        # Bound win probability
        win_prob_bat_1st = max(10.0, min(90.0, win_prob_bat_1st))
            
        par_score += score_modifier
        
        # Format par score as a range for more realism
        par_range = f"{int(par_score - (par_score*0.05))} - {int(par_score + (par_score*0.05))}"
        
        # Toss Game-Theory Matrix
        matrix = {
            "bat_first": {
                "expected_win_prob": f"{win_prob_bat_1st}%",
                "advantage": "Set the pace, avoid batting last on deteriorating pitch" if win_prob_bat_1st > 50 else "High risk of early collapse"
            },
            "bowl_first": {
                "expected_win_prob": f"{100 - win_prob_bat_1st}%",
                "advantage": "Exploit early moisture and movement" if win_prob_bat_1st < 50 else "Will face heavy run chase pressure"
            }
        }
        
        # 4. Spatio-Temporal Markovian Pitch Degradation Algorithm (ST-MPDA)
        import numpy as np
        x = np.linspace(-1.5, 1.5, 30)
        y = np.linspace(0, 20.12, 50)
        X, Y = np.meshgrid(x, y)
        
        phases = {}
        if match_format == "T20":
            phase_keys = ["Overs 1-6 (Powerplay)", "Overs 7-15 (Middle)", "Overs 16-20 (Death)"]
            wear_multipliers = [0.2, 0.6, 1.0]
        elif match_format == "ODI":
            phase_keys = ["Overs 1-10", "Overs 11-25", "Overs 26-40", "Overs 41-50"]
            wear_multipliers = [0.1, 0.4, 0.7, 1.0]
        else:
            phase_keys = ["Day 1 (Fresh)", "Day 2 (Settling)", "Day 3 (Wear Begins)", "Day 4 (Cracks Open)", "Day 5 (Dustbowl)"]
            wear_multipliers = [0.0, 0.25, 0.5, 0.8, 1.2]
            
        temporal_degradation_matrix = {}
        
        for i, p_key in enumerate(phase_keys):
            mult = wear_multipliers[i]
            
            if wear_heatmap_type == "green_seaming":
                # Grass wears off slightly, minor divots
                Z = np.sin(Y) * np.cos(X) * (0.1 + mult * 0.15)
                colorscale = "Greens"
            elif wear_heatmap_type == "dusty_spinning":
                # Cracks open up drastically over time
                Z = np.sin(Y*2) * np.cos(X*2) * (0.2 + mult * 0.8)
                colorscale = "YlOrBr"
            elif wear_heatmap_type == "damp_sticky":
                # Dries up and forms uneven ridges
                Z = np.sin(Y/2) * np.cos(X/2) * (0.5 - mult * 0.2)
                colorscale = "Blues"
            else:
                # Flat belter stays mostly flat but gets scuffed
                Z = (np.sin(Y*3) * np.cos(X*3)) * (mult * 0.1)
                colorscale = "Greys"
                
            temporal_degradation_matrix[p_key] = {
                "Z_grid": Z.tolist(),
                "colorscale": colorscale
            }
        
        return {
            "status": "success",
            "detected_nature": pitch_type.title() if pitch_type else "Standard",
            "par_score": par_range,
            "toss_decision": toss_decision,
            "game_theory_matrix": matrix,
            "optimal_bowling_length": optimal_length,
            "temporal_degradation_matrix": temporal_degradation_matrix
        }

pitch_engine = PitchIntelligenceEngine()
