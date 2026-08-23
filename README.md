# Context-Aware Adaptive Cricket Ball Intelligence System (Dual Architecture)

This project has evolved into a massive **Dual-Architecture Framework**. It contains two distinct, but incredibly powerful, halves:

1. **The Core Research Framework (The Main Project)**: A heavy, offline Machine Learning pipeline designed to parse millions of data points (HawkeyeStats, CricSheet, Weather APIs) to run physics simulations, anomaly detection, and ablation studies.
2. **The Live Broadcast Companion (The Chrome Extension)**: A real-time predictive engine that sits as an overlay on your browser during live matches. It uses Computer Vision and our custom ST-MPDA algorithm to predict the *very next ball* as you watch.

---

## 🏗️ Architecture 1: The Core Research Framework (Main Project)
This is the foundational brain of the project. It focuses on large-scale trajectory analysis and answering the question: *What should this delivery have looked like under current conditions, and what actually happened?*

- **Data Ingestion Pipeline**: Scrapes and merges Hawkeye stats, CricSheet match JSONs, and Open-Meteo weather data into a massive `master_dataset.parquet`.
- **Feature Engineering**: Tracks evolving ball-state (scuffing over time) and bowler career profiles.
- **Physics Models**: Uses XGBoost to simulate exact ball trajectories.
- **Anomaly Detection**: Flags unusual deliveries by computing the difference between expected and observed physics behavior.
- **Evaluation & SHAP**: Runs ablation studies and SHAP explainability plots for research analysis.

**To run the Main Project:**
```bash
python run_pipeline.py                    # Full pipeline (CricSheet + Weather)
python run_pipeline.py --models-only      # Skip data prep, run models only
python run_pipeline.py --dashboard        # Launch the Streamlit dashboard
```

---

## 🚀 Architecture 2: The Live Broadcast Companion (v2.0)
We built a real-time front-end on top of the backend intelligence to serve predictions instantly while you watch live matches (e.g., on Cricbuzz).

- **Chrome Extension Overlay**: Injects a sleek UI directly into the live scorecard (`content.js`, `sidebar.css`).
- **FastAPI Backend (`main.py`)**: A real-time API server that handles extension requests.
- **Computer Vision & ST-MPDA (`vision.py`, `pitch_analyzer.py`)**: Dynamically analyzes live broadcast frames to estimate pitch wear, Par Scores, and Toss Game Theory matrices using the Spatio-Temporal Markovian Pitch Degradation Algorithm.
- **Live Online Learning**: The AI actively "watches" the game. Every time a ball is bowled, it compares its prediction against the live commentary and automatically appends the outcome to `historical_training_data.csv`, meaning the model trains itself in real-time.

**To run the Live Companion:**
1. **Start the AI Backend**: `python -m src.api.main`
2. **Install Extension**: Load the `src/extension/` folder in `chrome://extensions/`.
3. Open Cricbuzz, hit **ENABLE VISION**, and watch the AI predict the next ball!
