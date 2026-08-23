import json

class PitchIntelligenceEngine:
    def __init__(self):
        self.base_par_score_t20 = 165
        self.base_par_score_test = 320
        
    def analyze(self, pitch_type: str, pitch_report_text: str, match_format: str = "T20"):
        """
        Analyzes the Pitch conditions (from CV and Text) to generate a Game Theory Matrix.
        """
        pitch_type = pitch_type.lower()
        pitch_report_text = pitch_report_text.lower()
        
        # 1. Determine Pitch Nature
        is_green = "green" in pitch_type or "seam" in pitch_type or "grass" in pitch_report_text or "swing" in pitch_report_text
        is_dry = "dry" in pitch_type or "dust" in pitch_type or "spin" in pitch_report_text or "cracks" in pitch_report_text
        is_damp = "damp" in pitch_type or "sticky" in pitch_report_text
        
        # 2. Base Adjustments
        par_score = self.base_par_score_t20 if match_format == "T20" else self.base_par_score_test
        toss_decision = "Bat First"
        win_prob_bat_1st = 55.0
        
        wear_heatmap_type = "standard"
        optimal_length = "Good Length (6m-8m)"
        
        if is_green:
            par_score -= (20 if match_format == "T20" else 45)
            toss_decision = "Bowl First"
            win_prob_bat_1st = 38.5
            wear_heatmap_type = "green_seaming"
            optimal_length = "Full (4m-6m) to invite the drive and induce edges."
            
        elif is_dry:
            par_score -= (10 if match_format == "T20" else 20)
            toss_decision = "Bat First"
            win_prob_bat_1st = 68.2
            wear_heatmap_type = "dusty_spinning"
            optimal_length = "Hard Length (8m) to extract uneven bounce."
            
        elif is_damp:
            par_score -= (30 if match_format == "T20" else 60)
            toss_decision = "Bowl First"
            win_prob_bat_1st = 32.0
            wear_heatmap_type = "damp_sticky"
            optimal_length = "Back of a Length (8m-10m) to let the ball misbehave off the sticky surface."
            
        else:
            # Belter
            par_score += (25 if match_format == "T20" else 80)
            toss_decision = "Bat First"
            win_prob_bat_1st = 60.0
            wear_heatmap_type = "flat_belter"
            optimal_length = "Yorkers and Wide Lines. Pitch offers nothing, rely on variations."
            
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
        
        return {
            "status": "success",
            "detected_nature": pitch_type.title() if pitch_type else "Standard",
            "par_score": par_score,
            "toss_decision": toss_decision,
            "game_theory_matrix": matrix,
            "optimal_bowling_length": optimal_length,
            "wear_heatmap": wear_heatmap_type
        }

pitch_engine = PitchIntelligenceEngine()
