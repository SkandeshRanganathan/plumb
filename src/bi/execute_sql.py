import duckdb
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SQL_DIR = ROOT_DIR / "sql" / "analytics"
BI_DATA_DIR = ROOT_DIR / "data" / "bi"
REPORTS_DIR = ROOT_DIR / "data" / "reports"

def execute_analytics():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize DuckDB and map our parquet files to virtual tables!
    con = duckdb.connect(database=':memory:')
    
    # Create a schema alias so our SQL queries work exactly as written
    con.execute(f"""
        CREATE SCHEMA IF NOT EXISTS bi;
        CREATE VIEW bi.fact_deliveries AS SELECT * FROM read_parquet('{BI_DATA_DIR.as_posix()}/fact_deliveries.parquet');
        CREATE VIEW bi.fact_anomalies AS SELECT * FROM read_parquet('{BI_DATA_DIR.as_posix()}/fact_anomalies.parquet');
        CREATE VIEW bi.dim_bowler AS SELECT * FROM read_parquet('{BI_DATA_DIR.as_posix()}/dim_bowler.parquet');
        CREATE VIEW bi.dim_match AS SELECT * FROM read_parquet('{BI_DATA_DIR.as_posix()}/dim_match.parquet');
    """)

    print("Executing SQL Analytical Queries...")

    # Run 01_bowler_analysis.sql
    with open(SQL_DIR / "01_bowler_analysis.sql", "r") as f:
        query_1 = f.read()
        print("Running Bowler Analysis...")
        df_bowlers = con.execute(query_1).df()
        df_bowlers.to_csv(REPORTS_DIR / "bowler_analysis_report.csv", index=False)
        print(f"Top 5 Bowlers by Speed:\n{df_bowlers.head(5)}\n")

    # Run 02_anomaly_analysis.sql
    with open(SQL_DIR / "02_anomaly_analysis.sql", "r") as f:
        query_2 = f.read()
        print("Running Venue Anomaly Analysis...")
        df_anomalies = con.execute(query_2).df()
        df_anomalies.to_csv(REPORTS_DIR / "venue_anomaly_report.csv", index=False)
        print(f"Top 5 Venues for Anomalies:\n{df_anomalies.head(5)}\n")

    print(f"SQL execution complete! Reports saved to {REPORTS_DIR}")

if __name__ == "__main__":
    execute_analytics()
