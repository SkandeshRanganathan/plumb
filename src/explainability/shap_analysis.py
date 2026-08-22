"""
shap_analysis.py  –  MODULE 13
SHAP (SHapley Additive exPlanations) Interpretability & Attribution Module.

Features:
1. Load saved XGBoost / Tree models from MODELS_SAVED directory.
2. Compute TreeExplainer SHAP values across test/sample delivery datasets.
3. Generate global summary bar plots and beeswarm plots saved to EXPERIMENTS/plots/shap_summary.png.
4. Provide delivery-level explanation via explain_delivery(delivery_row, model_bundle).
5. Generate human-readable narrative text explanations:
   'Major model-attributed contributors: 1. ball_state (contributes X cm) 2. wind_speed...'
6. Full unit conversions (metres to centimetres) and signed directional attribution.
"""

import sys
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Set project root and paths
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from config import (
    DATA_MASTER, DATA_PROCESSED, MODELS_SAVED, EXPERIMENTS, RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Ensure output directories exist
MODELS_SAVED.mkdir(parents=True, exist_ok=True)
EXPERIMENTS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = EXPERIMENTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = EXPERIMENTS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# 1. Model Loading & Discovery Helpers
# ────────────────────────────────────────────────────────────────────────────

def discover_saved_models(models_dir: Optional[Path] = None) -> List[Path]:
    """
    Search MODELS_SAVED directory for serialized XGBoost / ML model pkl files.

    Args:
        models_dir: Directory to scan (default: config.MODELS_SAVED).

    Returns:
        List of Path objects for discovered .pkl files.
    """
    search_dir = models_dir or MODELS_SAVED
    if not search_dir.exists():
        logger.warning(f"Models directory not found: {search_dir}")
        return []

    model_files = list(search_dir.glob("*.pkl")) + list(search_dir.glob("*.pickle"))
    logger.info(f"Discovered {len(model_files)} saved model files in {search_dir}")
    return sorted(model_files)


def load_saved_model(model_path_or_name: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a serialized model pkl file and return a standardized bundle dictionary.

    Args:
        model_path_or_name: Direct path or filename located in MODELS_SAVED.

    Returns:
        Standardized bundle dict:
          {
              "model": estimator_object,
              "features": list_of_feature_names,
              "target": target_name,
              "model_type": str,
              "filepath": Path
          }
    """
    path = Path(model_path_or_name)
    if not path.is_file():
        path = MODELS_SAVED / model_path_or_name
    if not path.is_file() and not path.suffix:
        path = MODELS_SAVED / f"{model_path_or_name}.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        loaded = pickle.load(f)

    # Standardize dictionary structure
    if isinstance(loaded, dict):
        model = loaded.get("model") or loaded.get("xgb_model") or loaded.get("pipeline") or loaded
        features = loaded.get("features") or loaded.get("feature_names") or []
        target = loaded.get("target") or loaded.get("target_name") or path.stem
        model_type = loaded.get("model_type") or type(model).__name__
    else:
        model = loaded
        features = getattr(model, "feature_names_in_", None)
        if features is not None:
            features = list(features)
        else:
            try:
                features = model.get_booster().feature_names
            except Exception:
                features = []
        target = path.stem
        model_type = type(model).__name__

    return {
        "model": model,
        "features": list(features) if features is not None else [],
        "target": str(target),
        "model_type": str(model_type),
        "filepath": path,
    }


# ────────────────────────────────────────────────────────────────────────────
# 2. SHAP Explainer Pipeline Class
# ────────────────────────────────────────────────────────────────────────────

class ShapExplainerPipeline:
    """
    SHAP TreeExplainer pipeline for computing local and global attributions
    for cricket trajectory, wide ball, and physics-residual models.
    """

    def __init__(
        self,
        model_bundle: Union[Dict[str, Any], Any],
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize explainer with model bundle or fitted model.

        Args:
            model_bundle: Standard bundle dict or raw fitted model object.
            feature_names: Optional explicit list of feature names.
        """
        if not _SHAP_AVAILABLE:
            raise ImportError(
                "SHAP library is required for ShapExplainerPipeline. Install via 'pip install shap'."
            )

        if isinstance(model_bundle, dict):
            self.model = model_bundle.get("model", model_bundle)
            self.features = model_bundle.get("features") or feature_names or []
            self.target_name = model_bundle.get("target", "prediction")
        else:
            self.model = model_bundle
            self.features = feature_names or getattr(model_bundle, "feature_names_in_", [])
            self.target_name = "prediction"

        # Initialize SHAP TreeExplainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Initialized shap.TreeExplainer successfully.")
        except Exception as e:
            logger.warning(f"TreeExplainer failed ({e}); falling back to generic Explainer.")
            self.explainer = shap.Explainer(self.model)

    def prepare_feature_dataframe(
        self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], pd.Series]
    ) -> pd.DataFrame:
        """
        Align and convert input data into an appropriately formatted DataFrame.

        Args:
            X: Input features.

        Returns:
            pd.DataFrame matching expected model features.
        """
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            df = pd.DataFrame([X])
        elif isinstance(X, np.ndarray):
            cols = self.features if len(self.features) == X.shape[1] else None
            df = pd.DataFrame(X, columns=cols)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Unsupported data format for SHAP computation: {type(X)}")

        # Reindex/align columns if features are specified
        if self.features:
            missing_cols = [c for c in self.features if c not in df.columns]
            for c in missing_cols:
                df[c] = 0.0  # fill missing one-hot or numeric features with 0
            df = df[self.features]

        return df

    def compute_shap_values(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, float, pd.DataFrame]:
        """
        Compute SHAP values for the given dataset.

        Args:
            X: Input feature matrix or DataFrame.

        Returns:
            Tuple of (shap_values_array, expected_base_value, X_df)
        """
        X_df = self.prepare_feature_dataframe(X)
        shap_res = self.explainer(X_df)

        if hasattr(shap_res, "values"):
            shap_values = shap_res.values
            base_value = float(
                np.mean(shap_res.base_values)
                if hasattr(shap_res.base_values, "__len__")
                else shap_res.base_values
            )
        else:
            shap_values = np.array(shap_res)
            base_val = getattr(self.explainer, "expected_value", 0.0)
            base_value = float(base_val if not hasattr(base_val, "__len__") else base_val[0])

        return shap_values, base_value, X_df

    def generate_summary_plot(
        self,
        X_sample: Union[pd.DataFrame, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        plot_type: str = "bar",
        max_display: int = 15,
        title: Optional[str] = None,
    ) -> Path:
        """
        Generate and save SHAP summary plot (bar or beeswarm).

        Args:
            X_sample: Dataset sample to compute SHAP values on.
            output_path: Destination file path (default: EXPERIMENTS/plots/shap_summary.png).
            plot_type: "bar" or "dot" (beeswarm).
            max_display: Number of top features to show.
            title: Optional custom plot title.

        Returns:
            Path to saved figure.
        """
        out_file = Path(output_path) if output_path else (PLOTS_DIR / "shap_summary.png")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        shap_values, base_val, X_df = self.compute_shap_values(X_sample)

        plt.figure(figsize=(10, 6), dpi=300)
        shap.summary_plot(
            shap_values,
            X_df,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
        )

        plot_title = title or f"SHAP Feature Importance ({self.target_name})"
        plt.title(plot_title, fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close("all")

        logger.info(f"Saved SHAP summary plot to: {out_file}")
        return out_file

    def generate_dependence_plot(
        self,
        feature_name: str,
        X_sample: Union[pd.DataFrame, np.ndarray],
        interaction_feature: Optional[str] = "auto",
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Generate and save SHAP dependence scatter plot for a specific feature.

        Args:
            feature_name: Feature to plot on x-axis.
            X_sample: Feature data matrix.
            interaction_feature: Interaction feature for coloring (or 'auto').
            output_path: Output PNG path.

        Returns:
            Path to saved plot.
        """
        out_file = (
            Path(output_path)
            if output_path
            else (PLOTS_DIR / f"shap_dependence_{feature_name}.png")
        )
        shap_values, _, X_df = self.compute_shap_values(X_sample)

        if feature_name not in X_df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in dataset columns.")

        plt.figure(figsize=(9, 5), dpi=300)
        shap.dependence_plot(
            feature_name,
            shap_values,
            X_df,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.title(f"SHAP Dependence: {feature_name}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close("all")

        logger.info(f"Saved SHAP dependence plot to: {out_file}")
        return out_file


# ────────────────────────────────────────────────────────────────────────────
# 3. Delivery Explanation & Narrative Text Generator
# ────────────────────────────────────────────────────────────────────────────

def explain_delivery(
    delivery_row: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
    model_bundle: Union[Dict[str, Any], Any],
    top_k: int = 5,
    unit: str = "cm",
) -> Dict[str, Any]:
    """
    Compute SHAP attribution for an individual delivery and extract top positive
    and negative contributors.

    Args:
        delivery_row: Series, dict, or 1-row DataFrame representing the delivery.
        model_bundle: Dictionary bundle or fitted tree model.
        top_k: Number of major contributors to extract.
        unit: Target measurement unit for explanation ('cm' or 'm').

    Returns:
        Explanation dictionary containing base value, predicted value,
        sorted feature contributions, top contributors, and a human-readable text explanation.
    """
    explainer_pipe = ShapExplainerPipeline(model_bundle)
    shap_values, base_value, X_df = explainer_pipe.compute_shap_values(delivery_row)

    # 1D arrays for single instance
    shap_1d = shap_values[0] if shap_values.ndim > 1 else shap_values
    feat_values_1d = X_df.iloc[0].values
    feat_names = list(X_df.columns)

    # Unit multiplier (if target is in metres, 1.0 m = 100.0 cm)
    scale_factor = 100.0 if unit.lower() == "cm" else 1.0
    unit_label = "cm" if unit.lower() == "cm" else "m"

    # Compute model predicted output
    model = explainer_pipe.model
    if hasattr(model, "predict"):
        try:
            pred_raw = float(model.predict(X_df)[0])
        except Exception:
            pred_raw = float(base_value + np.sum(shap_1d))
    else:
        pred_raw = float(base_value + np.sum(shap_1d))

    # Build feature contribution items
    contributions = []
    for name, s_val, f_val in zip(feat_names, shap_1d, feat_values_1d):
        impact_unit = float(s_val * scale_factor)
        direction = "increases" if s_val >= 0 else "decreases"
        contributions.append({
            "feature": name,
            "shap_value_raw": float(s_val),
            "shap_value_scaled": round(impact_unit, 2),
            "unit": unit_label,
            "feature_value": round(float(f_val), 3) if isinstance(f_val, (int, float, np.number)) else str(f_val),
            "direction": direction,
            "abs_impact": abs(impact_unit),
        })

    # Sort contributions by absolute impact descending
    contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

    # Separate positive and negative contributors
    top_positive = [c for c in contributions if c["shap_value_raw"] > 0][:top_k]
    top_negative = [c for c in contributions if c["shap_value_raw"] < 0][:top_k]
    top_overall = contributions[:top_k]

    result = {
        "target_name": explainer_pipe.target_name,
        "base_value_m": round(base_value, 4),
        "predicted_value_m": round(pred_raw, 4),
        "base_value_scaled": round(base_value * scale_factor, 2),
        "predicted_value_scaled": round(pred_raw * scale_factor, 2),
        "unit": unit_label,
        "top_features": top_overall,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "all_contributions": contributions,
    }

    # Generate narrative explanation text
    text_summary = generate_text_explanation(result, top_k=top_k, unit=unit_label)
    result["text_explanation"] = text_summary

    return result


def generate_text_explanation(
    explanation_dict: Dict[str, Any],
    top_k: int = 5,
    unit: str = "cm",
) -> str:
    """
    Format attribution dictionary into clean narrative text:
    'Major model-attributed contributors: 1. ball_state (contributes +4.2 cm) 2. wind_speed (-1.8 cm)...'

    Args:
        explanation_dict: Output dictionary from explain_delivery.
        top_k: Number of items to list in summary text.
        unit: Unit string.

    Returns:
        Formatted multi-line narrative string.
    """
    target = explanation_dict.get("target_name", "trajectory")
    base_val = explanation_dict.get("base_value_scaled", 0.0)
    pred_val = explanation_dict.get("predicted_value_scaled", 0.0)
    top_feats = explanation_dict.get("top_features", [])[:top_k]

    items = []
    for i, item in enumerate(top_feats, start=1):
        feat = item["feature"]
        val = item["shap_value_scaled"]
        f_val = item["feature_value"]
        sign = "+" if val >= 0 else ""
        items.append(f"{i}. {feat} (value={f_val}, contributes {sign}{val:.1f} {unit})")

    contributors_str = "\n   ".join(items) if items else "None"

    text = (
        f"Prediction Attribution for {target}:\n"
        f"  Baseline Expectation: {base_val:.2f} {unit}  ->  Model Prediction: {pred_val:.2f} {unit}\n"
        f"  Major model-attributed contributors:\n   {contributors_str}"
    )
    return text


# ────────────────────────────────────────────────────────────────────────────
# 4. Batch Pipeline Execution Runner
# ────────────────────────────────────────────────────────────────────────────

def run_shap_analysis_pipeline(
    sample_size: int = 500,
    model_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute SHAP explainability pipeline on saved models in MODELS_SAVED.
    Loads master dataset or creates representative sample, computes global SHAP summary
    plots, and generates sample delivery explanations.

    Args:
        sample_size: Maximum deliveries to sample for SHAP background evaluation.
        model_filter: Optional substring filter for model filenames (e.g. 'stumps_x').

    Returns:
        Dictionary summarizing generated artifacts and sample explanations.
    """
    print("=" * 70)
    print("SHAP EXPLAINABILITY & MODEL INTERPRETABILITY PIPELINE")
    print("=" * 70)

    if not _SHAP_AVAILABLE:
        print("[ERROR] 'shap' package is not installed. Please run: pip install shap")
        return {"error": "shap_not_installed"}

    discovered = discover_saved_models()
    if not discovered:
        print(f"[WARNING] No saved models found in {MODELS_SAVED}.")
        print("Please train models (e.g. trajectory_models.py) before running SHAP analysis.")
        return {"error": "no_models_found"}

    # Filter models if requested
    if model_filter:
        discovered = [m for m in discovered if model_filter in m.name]

    # Load master dataset for sample evaluation
    master_path = DATA_MASTER / "master_cricket_dataset.parquet"
    if master_path.exists():
        try:
            df_full = pd.read_parquet(master_path)
            logger.info(f"Loaded master dataset for SHAP evaluation: {len(df_full):,} rows")
        except Exception as e:
            logger.warning(f"Error loading master dataset: {e}. Creating synthetic fallback.")
            df_full = pd.DataFrame()
    else:
        df_full = pd.DataFrame()

    results_summary = {}

    for model_path in discovered:
        print(f"\n--- Analyzing Model: {model_path.name} ---")
        try:
            bundle = load_saved_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load {model_path}: {e}")
            continue

        model = bundle["model"]
        features = bundle["features"]

        if not features:
            print(f"  [SKIP] No feature names found for {model_path.name}")
            continue

        # Prepare dataset sample
        if not df_full.empty:
            avail_cols = [c for c in features if c in df_full.columns]
            sample_df = df_full[avail_cols].dropna().sample(
                n=min(sample_size, len(df_full)),
                random_state=RANDOM_SEED,
                replace=False if len(df_full) >= sample_size else True,
            )
        else:
            # Synthetic feature dataframe for testing/verification
            np.random.seed(RANDOM_SEED)
            synth_data = np.random.randn(min(sample_size, 100), len(features))
            sample_df = pd.DataFrame(synth_data, columns=features)

        try:
            explainer_pipe = ShapExplainerPipeline(bundle)

            # Generate global summary plot
            plot_name = f"shap_summary_{model_path.stem}.png"
            summary_plot_path = PLOTS_DIR / plot_name
            explainer_pipe.generate_summary_plot(
                sample_df,
                output_path=summary_plot_path,
                plot_type="bar",
                max_display=12,
                title=f"SHAP Feature Importance ({model_path.stem})",
            )
            print(f"  [OK] Summary plot saved: {summary_plot_path}")

            # Also generate canonical default shap_summary.png
            default_summary_path = PLOTS_DIR / "shap_summary.png"
            explainer_pipe.generate_summary_plot(
                sample_df,
                output_path=default_summary_path,
                plot_type="bar",
                max_display=15,
            )

            # Demonstrate single delivery explanation
            sample_delivery = sample_df.iloc[0]
            delivery_expl = explain_delivery(sample_delivery, bundle, top_k=5, unit="cm")

            print("\n  Sample Delivery Explanation:")
            print("  " + delivery_expl["text_explanation"].replace("\n", "\n  "))

            results_summary[model_path.stem] = {
                "summary_plot": str(summary_plot_path),
                "target": bundle["target"],
                "features_analyzed": len(features),
                "sample_delivery_explanation": delivery_expl["text_explanation"],
            }

        except Exception as e:
            logger.error(f"Error computing SHAP for {model_path.name}: {e}", exc_info=True)

    # Save summary report JSON
    summary_json_path = RESULTS_DIR / "shap_analysis_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[OK] SHAP analysis report saved to: {summary_json_path}")

    return results_summary


# ────────────────────────────────────────────────────────────────────────────
# 5. Main Entry Point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_shap_analysis_pipeline()
