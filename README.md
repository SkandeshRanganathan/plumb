# Context-Aware Adaptive Cricket Ball Intelligence System

A research-oriented AIML framework for understanding, modelling and predicting cricket-ball behaviour by combining ball-tracking data with **bowler characteristics, release geometry, pitch conditions, venue, weather, ball type, ball age and evolving ball-state information**.

The system is designed around one central question:

> **Can cricket-ball behaviour be predicted more accurately when the AI understands not only where the ball was released and where it travelled, but also who bowled it, how it was released, what state the ball was in, and the conditions under which it was delivered?**

Unlike a conventional ball-tracking system whose primary objective is to reconstruct the observed trajectory, this project focuses on building an **adaptive intelligence layer around the trajectory**.

The system attempts to estimate:

> **What should this delivery have looked like under the current conditions?**

and then compares that expectation with:

> **What actually happened?**

The difference between the expected and observed behaviour becomes a useful signal for trajectory analysis, unusual-delivery detection and downstream decision assistance.

---

# Project Structure

```text
cric/
├── run_pipeline.py              ← Master run script (start here)
├── src/
│   ├── config.py                ← All paths, bounds, constants
│   ├── ingestion/
│   │   ├── hawkeye_ingest.py    ← MODULE 1-A: Load + clean all 6 HawkeyeStats CSVs
│   │   ├── cricsheet_join.py    ← MODULE 1-D: Venue/date join from CricSheet
│   │   ├── weather_fetch.py     ← MODULE 1-E: Open-Meteo weather API
│   │   └── master_dataset.py    ← MODULE 1-H: Pipeline orchestrator
│   ├── features/
│   │   ├── bowler_profiles.py  ← MODULE 1-C: Bowler career profile features
│   │   └── ball_state.py       ← MODULE 1-G: Ball-state rolling features
│   ├── models/
│   │   ├── physics/
│   │   │   └── physics_model.py        ← MODULE 6: Physics trajectory simulation
│   │   ├── context_aware/
│   │   │   └── trajectory_models.py    ← MODULE 7: XGBoost context-aware models
│   │   ├── anomaly/
│   │   │   └── anomaly_detection.py    ← MODULE 10: Unusual delivery detection
│   │   ├── wide_ball/
│   │   │   └── wide_ball_model.py      ← MODULE 11: Wide-ball decision assistance
│   │   └── no_ball/
│   │       └── no_ball_model.py        ← MODULE 12: No-ball classification
│   ├── evaluation/
│   │   └── ablation_study.py   ← Experiments 1-8: Feature ablation
│   ├── explainability/
│   │   └── shap_analysis.py    ← MODULE 13: SHAP explainability
│   └── dashboard/
│       └── dashboard.py        ← MODULE 15: Streamlit research dashboard
├── data/
│   ├── processed/               ← Intermediate parquet files
│   ├── master/                  ← master_dataset.parquet (source of truth)
│   ├── external/
│   │   ├── cricsheet/           ← Downloaded CricSheet JSON files
│   │   └── weather/             ← Cached Open-Meteo responses
│   ├── bowler_profiles/         ← bowler_profiles.parquet + .csv
│   └── ball_state/              ← ball_state_summary.parquet
├── models/
│   └── saved/                   ← All saved .pkl model files
├── experiments/
│   ├── results/                 ← ablation_results.csv, trajectory_experiments.csv
│   └── plots/                   ← shap_*.png and other plots
└── datasets/                    ← Raw data (git-cloned repos)
    ├── hawkeye_stats/           ← 6 HawkeyeStats CSV files
    └── cric360/                 ← Cric360 broadcast images
