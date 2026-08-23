# Cricket AI: Live Broadcast Companion (v2.0)

## Where We Started
This project began as a static, research-oriented AIML framework called the *Context-Aware Adaptive Cricket Ball Intelligence System*. The original goal was to build a massive offline data pipeline processing HawkeyeStats, CricSheet, and Weather data through XGBoost physics models to perform anomaly detection and trajectory analysis (e.g. Ablation studies, SHAP explainability). 

## The Evolution
We fundamentally shifted the architecture from a static offline research pipeline into a **Live, Real-Time AI Companion**. 

Here is what we built to get here:
1. **Chrome Extension Frontend**: We developed a beautiful, professional overlay (`content.js`, `sidebar.css`) that injects directly into live scorecards like Cricbuzz.
2. **FastAPI & PostgreSQL Backend**: We ripped out the heavy offline batch processing and built a blazing-fast real-time API (`main.py`) to serve predictions instantaneously.
3. **Computer Vision & ST-MPDA**: We introduced `vision.py` to analyze broadcast frames, and the **ST-MPDA (Spatio-Temporal Markovian Pitch Degradation Algorithm)** in `pitch_analyzer.py` to calculate Par Scores, Pitch Wear, and Game Theory matrices dynamically.
4. **Historical Dataset Integration (v2.0)**: After running into ESPN/Akamai firewall blocks, we pivoted to a smarter approach. We built `historical_training_data.csv` to act as the model's historical memory, overriding simple heuristics with actual ML-driven Expected Wicket (xW) probabilities.
5. **Continuous Online Learning**: The AI actively "watches" the game with you. Every time a ball is bowled, it compares its prediction against the live commentary, extracts the actual outcome, and appends the data directly back into its dataset to get mathematically smarter as the match goes on.

## Where We Are Right Now
We have a fully functional **Live Predictive Engine**. 
Instead of analyzing spreadsheets from a database, you now have an extension that sits on your screen during a live match, tracks the pitch wear, estimates the Par Score, predicts the exact pace, angle, and delivery type of the *very next ball*, and visually tracks LBW trajectories in real-time. 

---

### Project Structure (Current Architecture)
```text
cric/
├── src/
│   ├── api/
│   │   ├── main.py                  ← FastAPI Backend (Prediction Engine & Online Learning)
│   │   ├── vision.py                ← Computer Vision broadcast analysis
│   │   ├── pitch_analyzer.py        ← ST-MPDA Algorithm for Par Scores & Pitch Wear
│   │   ├── historical_training_data.csv ← Core ML Dataset
│   │   └── dataset_generator.py     ← Historical crawler
│   ├── extension/
│   │   ├── manifest.json            ← Chrome Extension Manifest (v3)
│   │   ├── background.js            ← Service Worker for bypassing CORS
│   │   ├── content.js               ← DOM Injection & Live Polling Logic
│   │   └── sidebar.css              ← Styling for the UI
```

## How to Run

1. **Start the AI Backend**:
   ```bash
   python -m src.api.main
   ```
2. **Install the Extension**:
   - Go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked" and select the `src/extension` folder.
3. **Analyze Live**:
   - Open a live match on Cricbuzz.
   - Click the **ENABLE VISION** button to calculate Pitch conditions and Par Scores.
   - Watch the Next Ball Prediction engine learn and adapt dynamically!
