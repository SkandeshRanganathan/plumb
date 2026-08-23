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
        
        return {
            "status": "success",
            "detected_nature": pitch_type.title() if pitch_type else "Standard",
            "par_score": par_range,
            "toss_decision": toss_decision,
            "game_theory_matrix": matrix,
            "optimal_bowling_length": optimal_length,
            "wear_heatmap": wear_heatmap_type
        }

pitch_engine = PitchIntelligenceEngine()
