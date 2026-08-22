"""
physics_model.py  –  MODULE 6
Physics-based cricket ball trajectory model.

Models the ball trajectory as a physical projectile with:
  - Gravity (9.81 m/s²)
  - Aerodynamic drag (quadratic drag law)
  - Magnus lift (spin — approximated from bowling style)
  - Lateral seam/swing term (estimated from bowling style + conditions)

Given ONLY the release conditions (speed, approximate angle),
the model predicts:
  - Predicted bounce point: (pred_pitch_x, pred_pitch_y)
  - Predicted stump position: (pred_stumps_x, pred_stumps_y)

The RESIDUAL = Actual - Physics is the key research signal:
  residual_stumps_x = stumps_x - pred_stumps_x
  residual_stumps_y = stumps_y - pred_stumps_y

The residual captures everything the physics model cannot explain:
  - Swing deviation
  - Seam deviation
  - Post-bounce behaviour
  - Topspin/backspin effects
  - Pitch surface interaction

These residuals are then predicted by the context-aware ML model.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import PITCH_LENGTH_M


# ── Physical constants ────────────────────────────────────────────────────────
G          = 9.81       # m/s² gravity
RHO_STD    = 1.225      # kg/m³ air density at sea level, 15°C
BALL_MASS  = 0.1560     # kg (cricket ball: 155.9–163.0g, use midpoint)
BALL_RADIUS= 0.0359     # m (circumference ~22.4 cm)
BALL_AREA  = np.pi * BALL_RADIUS ** 2  # cross-sectional area m²

# Drag coefficient for a cricket ball (smooth vs seam side)
# Conventional swing: CD varies between 0.25 (attached flow) and 0.45 (turbulent)
CD_DEFAULT = 0.40       # typical worn ball
CD_NEW     = 0.28       # new ball (laminar on one side, turbulent other)
CD_OLD     = 0.45       # very old ball

# Magnus lift coefficient approximation for spin bowlers
CL_SPIN    = 0.25       # rough estimate for slow bowlers
CL_SEAM    = 0.05       # minimal Magnus for pure seam bowlers

# Approximate release height (metres above ground)
RELEASE_HEIGHT = 2.20   # height of bowler's hand at release

# Approximate release angle (degrees below horizontal for typical delivery)
RELEASE_ANGLE_DEFAULT = -3.0  # degrees (negative = slightly downward)

# Pitch geometry
PITCH_LENGTH = PITCH_LENGTH_M  # 20.12 m
STUMP_HEIGHT = 0.711          # m (top of bails)
STUMP_X_HALF = 0.1143         # half-width of stumps (22.86 cm total / 2)


# ── Helper functions ──────────────────────────────────────────────────────────

def rho_from_conditions(temperature_c: float = 15.0,
                        pressure_hpa: float = 1013.25,
                        humidity_pct: float = 50.0) -> float:
    """
    Compute air density (kg/m³) from meteorological conditions.
    Uses the ideal gas law approximation with humidity correction.
    """
    T_K   = temperature_c + 273.15
    P_Pa  = pressure_hpa * 100.0
    # Saturation vapour pressure (Buck equation approximation)
    P_sat = 611.2 * np.exp(17.62 * temperature_c / (243.12 + temperature_c))
    P_v   = (humidity_pct / 100.0) * P_sat   # partial pressure of water vapour
    P_d   = P_Pa - P_v                        # partial pressure of dry air
    R_d   = 287.058   # J/(kg·K) dry air
    R_v   = 461.495   # J/(kg·K) water vapour
    rho   = P_d / (R_d * T_K) + P_v / (R_v * T_K)
    return rho


def cd_from_ball_age(ball_age_overs: float) -> float:
    """Estimate drag coefficient from ball age (heuristic)."""
    if ball_age_overs < 10:
        return CD_NEW
    elif ball_age_overs > 60:
        return CD_OLD
    else:
        t = (ball_age_overs - 10) / 50.0
        return CD_NEW + t * (CD_OLD - CD_NEW)


def cl_from_style(bowling_style: str) -> float:
    """Estimate Magnus lift coefficient from bowling style."""
    if "SPIN" in str(bowling_style) or "ORTHODOX" in str(bowling_style) or \
       "UNORTHODOX" in str(bowling_style):
        return CL_SPIN
    return CL_SEAM


# ── Trajectory simulation ─────────────────────────────────────────────────────

@dataclass
class PhysicsTrajectory:
    """Result of a single physics trajectory simulation."""
    # Bounce point (metres from bowler's stumps to pitch impact)
    pred_pitch_x: float   # lateral
    pred_pitch_y: float   # length along pitch
    # Stump position at batter end
    pred_stumps_x: float  # lateral
    pred_stumps_y: float  # height
    # Intermediate values
    time_to_pitch: float   # seconds
    speed_at_pitch: float  # m/s
    valid: bool = True
    note: str = ""


def simulate_trajectory(
    speed_ms: float,
    bowling_style: str = "FAST_SEAM",
    lateral_offset: float = 0.0,      # release position lateral offset (m)
    ball_age_overs: float = 20.0,
    temperature_c: float = 25.0,
    pressure_hpa: float = 1013.0,
    humidity_pct: float = 60.0,
    swing_direction: float = 0.0,     # +1 = away from centre, -1 = toward centre
    dt: float = 0.002,                # integration time step (s)
    release_x: float = 0.0,           # lateral release position
    release_z: float = 2.20,          # release height
) -> PhysicsTrajectory:
    """
    Simulate cricket ball trajectory using Euler integration of equations of motion.

    Coordinate system:
      x = lateral (off-stump direction for RHB)
      y = along pitch (0 = bowler's stumps, ~20.12 = batter's stumps)
      z = height (0 = ground)

    Returns predicted bounce point and stump crossing position.
    """
    if speed_ms <= 0 or np.isnan(speed_ms):
        return PhysicsTrajectory(
            pred_pitch_x=0.0, pred_pitch_y=9.0,
            pred_stumps_x=0.0, pred_stumps_y=0.72,
            time_to_pitch=0.5, speed_at_pitch=speed_ms,
            valid=False, note="invalid speed"
        )

    # Air density
    rho = rho_from_conditions(temperature_c, pressure_hpa, humidity_pct)

    # Ball coefficients
    Cd = cd_from_ball_age(ball_age_overs)
    Cl = cl_from_style(bowling_style)
    k  = 0.5 * rho * BALL_AREA / BALL_MASS   # drag/lift prefactor

    # Initial conditions
    release_angle_rad = np.radians(RELEASE_ANGLE_DEFAULT)
    # Decompose speed into components
    vz0 = speed_ms * np.sin(release_angle_rad)   # small downward component
    vy0 = speed_ms * np.cos(release_angle_rad) * np.cos(np.radians(1.0))
    
    # Calculate the initial horizontal angle to aim at pitch_x
    dx = lateral_offset - release_x
    dy = PITCH_LENGTH * 0.5  # estimate pitch is halfway down
    aim_angle_rad = np.arctan2(dx, dy)
    vx0 = speed_ms * np.sin(aim_angle_rad) + (swing_direction * speed_ms * 0.01)

    pos = np.array([release_x, 0.0, release_z], dtype=float)
    vel = np.array([vx0, vy0, vz0], dtype=float)

    bounce_recorded = False
    pitch_x, pitch_y, time_pitch, speed_pitch = None, None, None, None
    stumps_x, stumps_y = None, None

    max_steps = int(2.0 / dt)   # simulate up to 2 seconds

    for step in range(max_steps):
        speed_cur = np.linalg.norm(vel)
        if speed_cur < 1e-6:
            break

        # Drag force (opposing velocity)
        drag = -k * Cd * speed_cur * vel

        # Magnus force (simplified: acts laterally for seam/swing)
        # For swing bowling: force acts laterally (x-axis)
        magnus_x = k * Cl * swing_direction * speed_cur ** 2

        # Gravity (z-direction only)
        accel = np.array([
            drag[0] + magnus_x,   # x: drag + swing/Magnus
            drag[1],               # y: drag only (along pitch)
            drag[2] - G            # z: drag + gravity
        ])

        vel += accel * dt
        pos += vel * dt

        # Check bounce (z ≤ 0 and still traveling forward)
        if not bounce_recorded and pos[2] <= 0.0 and pos[1] > 1.0:
            bounce_recorded = True
            pitch_x   = float(pos[0])
            pitch_y   = float(pos[1])
            time_pitch = float(step * dt)
            speed_pitch = float(np.linalg.norm(vel))

            # Simple bounce model: dampen z velocity, reduce x/y slightly
            vel[2] = -vel[2] * 0.55    # coefficient of restitution ~0.55
            vel[0] *= 0.80             # lateral friction
            vel[1] *= 0.90             # longitudinal friction
            pos[2]  = 0.001            # reset to ground level

        # Check stump crossing (y ≥ PITCH_LENGTH)
        if pos[1] >= PITCH_LENGTH:
            stumps_x = float(pos[0])
            stumps_y = float(pos[2])
            break

    # Fallback if simulation didn't reach stumps
    if stumps_x is None:
        stumps_x = pos[0]
        stumps_y = max(0.0, pos[2])
    if pitch_x is None:
        # Ball never bounced (full toss)
        pitch_x, pitch_y = lateral_offset, PITCH_LENGTH * 0.9
        time_pitch  = 0.5
        speed_pitch = speed_ms * 0.95

    return PhysicsTrajectory(
        pred_pitch_x  = round(pitch_x, 4),
        pred_pitch_y  = round(max(pitch_y, 0.0), 4),
        pred_stumps_x = round(stumps_x, 4),
        pred_stumps_y = round(max(stumps_y, 0.0), 4),
        time_to_pitch = round(time_pitch, 4),
        speed_at_pitch= round(speed_pitch, 4),
        valid         = True,
    )


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_physics_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply physics simulation to all rows in the DataFrame.
    Requires: ball_speed_ms, bowling_style, lateral_swing (for swing direction),
              ball_age_overs, temperature_c, pressure_hpa, humidity_pct

    Adds columns:
      phys_pred_pitch_x, phys_pred_pitch_y
      phys_pred_stumps_x, phys_pred_stumps_y
      phys_time_to_pitch, phys_speed_at_pitch, phys_valid
      residual_pitch_x, residual_pitch_y
      residual_stumps_x, residual_stumps_y
    """
    print("  Running physics trajectory simulation ...")
    results = []

    for idx, row in df.iterrows():
        speed = row.get("ball_speed_ms")
        style = row.get("bowling_style", "FAST_SEAM")
        lateral = row.get("pitch_x")
        lateral = float(lateral) if pd.notna(lateral) else 0.0
        age = row.get("ball_age_overs")
        age = float(age) if pd.notna(age) else 20.0
        temp = row.get("temperature_c")
        temp = float(temp) if pd.notna(temp) else 25.0
        pres = row.get("pressure_hpa")
        pres = float(pres) if pd.notna(pres) else 1013.0
        hum  = row.get("humidity_pct")
        hum = float(hum) if pd.notna(hum) else 60.0

        # Infer swing direction from bowling style and handedness
        style_str = str(style)
        swing_dir = 0.0
        if "FAST" in style_str or "MEDIUM" in style_str:
            # New ball: away swing; old ball: reverse
            if age < 25:
                swing_dir = 1.0   # away swing
            elif age > 55:
                swing_dir = -1.0  # reverse swing
        elif "SPIN" in style_str or "ORTHODOX" in style_str:
            swing_dir = 0.5  # some lateral drift from spin

        traj = simulate_trajectory(
            speed_ms=float(speed) if pd.notna(speed) else 35.0,
            bowling_style=style_str,
            lateral_offset=float(lateral),
            ball_age_overs=float(age),
            temperature_c=float(temp),
            pressure_hpa=float(pres),
            humidity_pct=float(hum),
            swing_direction=swing_dir,
            release_x=0.0,  # Default or extract from df if available
            release_z=2.20  # Default or extract from df if available
        )
        results.append({
            "phys_pred_pitch_x":  traj.pred_pitch_x,
            "phys_pred_pitch_y":  traj.pred_pitch_y,
            "phys_pred_stumps_x": traj.pred_stumps_x,
            "phys_pred_stumps_y": traj.pred_stumps_y,
            "phys_valid":         int(traj.valid),
        })

    phys_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, phys_df], axis=1)

    # Compute residuals (only where actual and physics are both available)
    df["residual_pitch_x"]  = np.where(df["pitch_x"].notna()  & df["phys_valid"]==1,
                                        df["pitch_x"]  - df["phys_pred_pitch_x"],  np.nan)
    df["residual_pitch_y"]  = np.where(df["pitch_y"].notna()  & df["phys_valid"]==1,
                                        df["pitch_y"]  - df["phys_pred_pitch_y"],  np.nan)
    df["residual_stumps_x"] = np.where(df["stumps_x"].notna() & df["phys_valid"]==1,
                                        df["stumps_x"] - df["phys_pred_stumps_x"], np.nan)
    df["residual_stumps_y"] = np.where(df["stumps_y"].notna() & df["phys_valid"]==1,
                                        df["stumps_y"] - df["phys_pred_stumps_y"], np.nan)

    valid = df["phys_valid"] == 1
    print(f"  ✓ Physics predictions: {valid.sum():,} valid rows")
    print(f"  Mean |residual_stumps_x|: {df.loc[valid,'residual_stumps_x'].abs().mean():.4f} m")
    print(f"  Mean |residual_stumps_y|: {df.loc[valid,'residual_stumps_y'].abs().mean():.4f} m")

    return df


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_physics_model(df: pd.DataFrame) -> dict:
    """Compute error metrics for the physics model on rows with valid data."""
    valid = df[(df["phys_valid"] == 1) &
               df["stumps_x"].notna() & df["stumps_y"].notna()]
    if len(valid) == 0:
        return {}

    def mae(col): return valid[col].abs().mean()
    def rmse(col): return np.sqrt((valid[col] ** 2).mean())

    return {
        "n_valid":              len(valid),
        "mae_stumps_x":         round(mae("residual_stumps_x"), 4),
        "rmse_stumps_x":        round(rmse("residual_stumps_x"), 4),
        "mae_stumps_y":         round(mae("residual_stumps_y"), 4),
        "rmse_stumps_y":        round(rmse("residual_stumps_y"), 4),
        "mae_pitch_x":          round(mae("residual_pitch_x"), 4),
        "rmse_pitch_x":         round(rmse("residual_pitch_x"), 4),
        "mae_pitch_y":          round(mae("residual_pitch_y"), 4),
        "rmse_pitch_y":         round(rmse("residual_pitch_y"), 4),
        "euclidean_stumps_rmse": round(
            np.sqrt(((valid["residual_stumps_x"]**2 +
                      valid["residual_stumps_y"]**2)).mean()), 4
        ),
    }


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 6: Physics Trajectory Model — Test Run")
    print("=" * 65)

    # Quick sanity test
    test_cases = [
        {"speed_ms": 40.0, "bowling_style": "FAST_SEAM",  "ball_age_overs": 5.0,  "label": "New ball fast"},
        {"speed_ms": 22.0, "bowling_style": "OFF_SPIN",   "ball_age_overs": 30.0, "label": "Offspinner"},
        {"speed_ms": 38.0, "bowling_style": "FAST_SEAM",  "ball_age_overs": 70.0, "label": "Old ball (reverse?)"},
        {"speed_ms": 35.0, "bowling_style": "MEDIUM_SEAM","ball_age_overs": 15.0, "label": "Medium pace"},
    ]
    for case in test_cases:
        label = case.pop("label")
        traj = simulate_trajectory(**case)
        print(f"\n{label}:")
        print(f"  Bounce: x={traj.pred_pitch_x:+.3f}m  y={traj.pred_pitch_y:.2f}m")
        print(f"  Stumps: x={traj.pred_stumps_x:+.3f}m  y={traj.pred_stumps_y:.3f}m")
        print(f"  Time to bounce: {traj.time_to_pitch:.3f}s  Speed at bounce: {traj.speed_at_pitch*3.6:.1f}km/h")
