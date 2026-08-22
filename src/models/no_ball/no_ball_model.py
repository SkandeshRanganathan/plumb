"""
no_ball_model.py  –  MODULE 12
Cricket No-Ball Intelligence & Detection Module.

Integrates dual capabilities:
1. Computer Vision Front-Crease / No-Ball Classifier (HOG + SVM):
   - Trained on the vibhudave ball-dataset (25 legal, 52 no-ball images).
   - Extracts Histogram of Oriented Gradients (HOG) features via scikit-image.
   - Evaluated via Leave-One-Out Cross-Validation (LOOCV) suitable for tiny datasets.
   - Saves model bundle to MODELS_SAVED / "no_ball_hog_svm.pkl".

2. Statistical Analysis of No-Ball Patterns from HawkeyeStats:
   - Analyzes extras 'Nb' annotations across all formats (IPL, ODI, Test).
   - Computes career no-ball rates, bowler-level vulnerability profiles,
     phase-of-play distributions (Powerplay/Middle/Death), and bowling style impacts.

LIMITATIONS & ETHICAL NOTES:
  * Dataset Size: 77 images total (25 legal, 52 no ball) is an extremely small,
    unrepresentative sample collected under varied camera angles and resolutions.
  * Overfitting Risk: High dimensional HOG descriptors on small sample size can overfit.
  * Production Deployment: Real-time cricket broadcasting officiating (MCC Law 21)
    requires synchronized high-speed (>= 500 fps) side-on crease cameras and calibrated
    popping-crease segmentation. This module serves as a research benchmark and baseline.
"""

import sys
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Set project root and paths
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

# Optional image processing imports with graceful handling
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from skimage.feature import hog
    from skimage.color import rgb2gray
    from skimage.transform import resize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

from config import (
    DATA_MASTER, DATA_PROCESSED, MODELS_SAVED, EXPERIMENTS,
    RANDOM_SEED, HAWKEYE_FILES, KAGGLE_CACHE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Ensure directories exist
MODELS_SAVED.mkdir(parents=True, exist_ok=True)
EXPERIMENTS.mkdir(parents=True, exist_ok=True)
(EXPERIMENTS / "results").mkdir(parents=True, exist_ok=True)
(EXPERIMENTS / "plots").mkdir(parents=True, exist_ok=True)

# Default path to vibhudave ball dataset
DEFAULT_VIBHUDAVE_DIR = (
    KAGGLE_CACHE / "vibhudave" / "ball-dataset" / "versions" / "1"
    if KAGGLE_CACHE.exists()
    else Path.home() / ".cache" / "kagglehub" / "datasets" / "vibhudave" / "ball-dataset" / "versions" / "1"
)


# ────────────────────────────────────────────────────────────────────────────
# 1. Image Loading and HOG Feature Extraction
# ────────────────────────────────────────────────────────────────────────────

class HOGFeatureExtractor:
    """
    Histogram of Oriented Gradients (HOG) feature extractor for crease/ball images.
    Resizes images to a standard dimension and extracts normalized gradient histograms.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        transform_sqrt: bool = True,
        block_norm: str = "L2-Hys",
    ):
        """
        Initialize HOG parameters.

        Args:
            image_size: Target (height, width) for resizing.
            orientations: Number of gradient orientation bins.
            pixels_per_cell: Size (in pixels) of each cell.
            cells_per_block: Number of cells in each block.
            transform_sqrt: Apply square-root power compression to normalize illumination.
            block_norm: Normalization method for blocks.
        """
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.transform_sqrt = transform_sqrt
        self.block_norm = block_norm

    def load_and_preprocess_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load an image, convert to grayscale, and resize.

        Args:
            image_path: Path to the image file.

        Returns:
            Normalized 2D float array (0.0 to 1.0) or None if load fails.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        try:
            if _PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    img = img.convert("L")  # Convert to grayscale
                    img = img.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    return arr
            elif _SKIMAGE_AVAILABLE:
                import skimage.io
                img = skimage.io.imread(str(image_path))
                if img.ndim == 3:
                    img = rgb2gray(img)
                img = resize(img, self.image_size, anti_aliasing=True)
                return img.astype(np.float32)
            else:
                raise ImportError("Neither PIL nor skimage is available for image loading.")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def extract_features(self, image_input: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract 1D HOG feature vector from an image path or array.

        Args:
            image_input: Filepath or preloaded numpy array.

        Returns:
            1D numpy array of HOG features or None on failure.
        """
        if isinstance(image_input, (str, Path)):
            img_arr = self.load_and_preprocess_image(image_input)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                if _SKIMAGE_AVAILABLE:
                    img_arr = rgb2gray(image_input)
                else:
                    img_arr = np.mean(image_input, axis=2) / 255.0
            else:
                img_arr = image_input.astype(np.float32)

            if img_arr.shape != self.image_size:
                if _SKIMAGE_AVAILABLE:
                    img_arr = resize(img_arr, self.image_size, anti_aliasing=True)
                elif _PIL_AVAILABLE:
                    pimg = Image.fromarray((img_arr * 255).astype(np.uint8))
                    pimg = pimg.resize((self.image_size[1], self.image_size[0]))
                    img_arr = np.array(pimg, dtype=np.float32) / 255.0
        else:
            logger.error(f"Unsupported image input type: {type(image_input)}")
            return None

        if img_arr is None:
            return None

        if _SKIMAGE_AVAILABLE:
            try:
                features = hog(
                    img_arr,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    transform_sqrt=self.transform_sqrt,
                    block_norm=self.block_norm,
                    feature_vector=True,
                )
                return features
            except Exception as e:
                logger.error(f"HOG computation error: {e}")
                return None
        else:
            # Fallback simple intensity and gradient descriptor if skimage is missing
            gx = np.gradient(img_arr, axis=1)
            gy = np.gradient(img_arr, axis=0)
            mag = np.sqrt(gx**2 + gy**2)
            downsampled = resize(mag, (16, 16)) if _SKIMAGE_AVAILABLE else mag[::8, ::8]
            return downsampled.flatten()


# ────────────────────────────────────────────────────────────────────────────
# 2. Dataset Loader for vibhudave ball-dataset
# ────────────────────────────────────────────────────────────────────────────

class NoBallDatasetLoader:
    """
    Loads images and labels from the vibhudave ball-dataset directory structure.
    Expected structure:
      <root>/legal ball/*.jpg  -> label 0
      <root>/no ball/*.jpg     -> label 1
    """

    def __init__(self, dataset_dir: Optional[Union[str, Path]] = None):
        """
        Initialize dataset loader.

        Args:
            dataset_dir: Root directory of vibhudave ball-dataset.
        """
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_VIBHUDAVE_DIR

    def load_dataset(self) -> Tuple[List[Path], np.ndarray, List[str]]:
        """
        Scan directory and return list of filepaths, target labels (0=legal, 1=no ball),
        and class names.

        Returns:
            Tuple of (file_paths, y_array, class_names)
        """
        file_paths = []
        labels = []

        if not self.dataset_dir.exists():
            logger.warning(f"Vibhudave dataset directory not found at: {self.dataset_dir}")
            return [], np.array([]), ["legal ball", "no ball"]

        legal_dir = self.dataset_dir / "legal ball"
        noball_dir = self.dataset_dir / "no ball"

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Load legal ball images (class 0)
        if legal_dir.exists():
            for p in legal_dir.iterdir():
                if p.is_file() and p.suffix.lower() in valid_exts:
                    file_paths.append(p)
                    labels.append(0)
        else:
            logger.warning(f"'legal ball' subfolder missing in {self.dataset_dir}")

        # Load no ball images (class 1)
        if noball_dir.exists():
            for p in noball_dir.iterdir():
                if p.is_file() and p.suffix.lower() in valid_exts:
                    file_paths.append(p)
                    labels.append(1)
        else:
            logger.warning(f"'no ball' subfolder missing in {self.dataset_dir}")

        logger.info(
            f"Found {len(file_paths)} total images: "
            f"{labels.count(0)} legal balls, {labels.count(1)} no balls."
        )

        return file_paths, np.array(labels, dtype=int), ["legal ball", "no ball"]


# ────────────────────────────────────────────────────────────────────────────
# 3. No-Ball Model Pipeline & Leave-One-Out Cross-Validation
# ────────────────────────────────────────────────────────────────────────────

class NoBallVisionClassifier:
    """
    HOG + SVM classifier pipeline for front-crease no-ball detection.
    Encapsulates feature extraction, scaling, SVM classification, and probability calibration.
    """

    def __init__(
        self,
        extractor: Optional[HOGFeatureExtractor] = None,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        random_state: int = RANDOM_SEED,
    ):
        self.extractor = extractor or HOGFeatureExtractor()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False
        self.cv_results_: Dict[str, Any] = {}

    def _build_pipeline(self) -> Pipeline:
        """Construct scikit-learn pipeline with StandardScaler and SVC."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                probability=True,
                class_weight="balanced",
                random_state=self.random_state,
            ))
        ])

    def extract_features_matrix(
        self, file_paths: List[Path], labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
        """
        Extract HOG features for a list of image paths.

        Returns:
            X (feature matrix), y (filtered labels), valid_paths (retained file paths)
        """
        X_list = []
        y_list = []
        valid_paths = []

        for p, y in zip(file_paths, labels):
            feat = self.extractor.extract_features(p)
            if feat is not None:
                X_list.append(feat)
                y_list.append(y)
                valid_paths.append(p)
            else:
                logger.warning(f"Could not extract features from {p}")

        if not X_list:
            return np.empty((0, 0)), np.array([]), []

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int), valid_paths

    def evaluate_leave_one_out(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform Leave-One-Out Cross-Validation (LOOCV).
        Essential for robust evaluation on n=77 small dataset.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary target array of shape (n_samples,).

        Returns:
            Dictionary containing metrics, predictions, probabilities, and confusion matrix.
        """
        if len(X) == 0 or len(y) == 0:
            logger.warning("Empty dataset provided for LOOCV.")
            return {}

        loo = LeaveOneOut()
        y_pred = np.zeros(len(y), dtype=int)
        y_prob = np.zeros(len(y), dtype=float)

        logger.info(f"Running Leave-One-Out Cross-Validation over {len(y)} samples...")

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe = self._build_pipeline()
            pipe.fit(X_train, y_train)

            pred = pipe.predict(X_test)[0]
            prob = pipe.predict_proba(X_test)[0, 1]

            y_pred[test_idx] = pred
            y_prob[test_idx] = prob

        # Metrics computation
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        # ROC AUC
        try:
            auc = roc_auc_score(y, y_prob)
        except Exception:
            auc = float("nan")

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results = {
            "n_samples": int(len(y)),
            "n_legal": int((y == 0).sum()),
            "n_no_ball": int((y == 1).sum()),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": {
                "true_negative_legal": int(tn),
                "false_positive_called_nb": int(fp),
                "false_negative_missed_nb": int(fn),
                "true_positive_nb": int(tp),
            },
            "y_true": y.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist(),
            "notes": (
                "Evaluated using LeaveOneOut Cross-Validation on the 77-image vibhudave ball-dataset. "
                "HOG feature dimension: {dim}. Model is experimental baseline."
            ).format(dim=X.shape[1] if X.ndim > 1 else 0),
        }

        self.cv_results_ = results
        return results

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NoBallVisionClassifier":
        """
        Fit final model on the full available dataset.

        Args:
            X: Feature matrix.
            y: Binary labels.
        """
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict_image(
        self, image_input: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Predict whether an input image represents a NO BALL or a LEGAL BALL.

        Args:
            image_input: Filepath or numpy image array.

        Returns:
            Dictionary with prediction decision, probabilities, confidence, and explanation.
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() or load_model().")

        feat = self.extractor.extract_features(image_input)
        if feat is None:
            return {
                "decision": "ERROR",
                "is_no_ball": None,
                "no_ball_probability": np.nan,
                "confidence": 0.0,
                "reason": "Failed to process image or extract HOG features."
            }

        feat_2d = feat.reshape(1, -1)
        pred_label = int(self.pipeline.predict(feat_2d)[0])
        prob_nb = float(self.pipeline.predict_proba(feat_2d)[0, 1])

        decision = "NO_BALL" if pred_label == 1 else "LEGAL"
        confidence = prob_nb if pred_label == 1 else (1.0 - prob_nb)

        return {
            "decision": decision,
            "is_no_ball": pred_label,
            "no_ball_probability": round(prob_nb, 4),
            "confidence": round(confidence, 4),
            "reason": (
                f"HOG+SVM classifier predicts {decision} with {confidence*100:.1f}% confidence "
                f"(P(NoBall) = {prob_nb:.3f})."
            )
        }

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Serialize model bundle to disk.

        Args:
            filepath: Destination path (default: MODELS_SAVED / 'no_ball_hog_svm.pkl').
        """
        out_path = Path(filepath) if filepath else (MODELS_SAVED / "no_ball_hog_svm.pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "pipeline": self.pipeline,
            "extractor_config": {
                "image_size": self.extractor.image_size,
                "orientations": self.extractor.orientations,
                "pixels_per_cell": self.extractor.pixels_per_cell,
                "cells_per_block": self.extractor.cells_per_block,
                "transform_sqrt": self.extractor.transform_sqrt,
                "block_norm": self.extractor.block_norm,
            },
            "cv_results": self.cv_results_,
            "model_type": "HOG_SVM_NoBall_Classifier",
            "dataset_info": "vibhudave/ball-dataset (77 images)",
            "limitations": (
                "Experimental model trained on 77 images. Not intended for direct live officiating."
            ),
        }

        with open(out_path, "wb") as f:
            pickle.dump(bundle, f)

        logger.info(f"Saved no-ball model bundle to: {out_path}")
        return out_path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "NoBallVisionClassifier":
        """
        Load serialized model bundle.

        Args:
            filepath: Path to pkl file.
        """
        in_path = Path(filepath) if filepath else (MODELS_SAVED / "no_ball_hog_svm.pkl")
        if not in_path.exists():
            raise FileNotFoundError(f"No saved model found at {in_path}")

        with open(in_path, "rb") as f:
            bundle = pickle.load(f)

        ext_cfg = bundle.get("extractor_config", {})
        extractor = HOGFeatureExtractor(
            image_size=ext_cfg.get("image_size", (128, 128)),
            orientations=ext_cfg.get("orientations", 9),
            pixels_per_cell=ext_cfg.get("pixels_per_cell", (8, 8)),
            cells_per_block=ext_cfg.get("cells_per_block", (2, 2)),
            transform_sqrt=ext_cfg.get("transform_sqrt", True),
            block_norm=ext_cfg.get("block_norm", "L2-Hys"),
        )

        instance = cls(extractor=extractor)
        instance.pipeline = bundle.get("pipeline")
        instance.cv_results_ = bundle.get("cv_results", {})
        instance.is_fitted = instance.pipeline is not None
        return instance


# ────────────────────────────────────────────────────────────────────────────
# 4. Statistical Analysis of No-Ball Patterns from HawkeyeStats
# ────────────────────────────────────────────────────────────────────────────

class HawkeyeNoBallAnalysis:
    """
    Statistical analyzer for no-ball incidence, bowler-level propensity,
    format rates, and match phase distributions from HawkeyeStats extras data.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize analyzer with DataFrame or load default data.
        """
        self.df = df if df is not None else self._load_hawkeye_data()

    def _load_hawkeye_data(self) -> pd.DataFrame:
        """Load data from master dataset or raw Hawkeye files."""
        # Try master dataset first
        master_path = DATA_MASTER / "master_cricket_dataset.parquet"
        if master_path.exists():
            try:
                df = pd.read_parquet(master_path)
                logger.info(f"Loaded master dataset for no-ball analysis: {len(df):,} rows")
                return df
            except Exception as e:
                logger.warning(f"Could not load master parquet: {e}")

        # Try unified processed data
        proc_path = DATA_PROCESSED / "hawkeye_unified.parquet"
        if proc_path.exists():
            try:
                df = pd.read_parquet(proc_path)
                logger.info(f"Loaded processed hawkeye dataset: {len(df):,} rows")
                return df
            except Exception as e:
                logger.warning(f"Could not load processed parquet: {e}")

        # Fallback to loading CSV files directly
        dfs = []
        for name, path in HAWKEYE_FILES.items():
            if path.exists():
                try:
                    sub = pd.read_csv(path, low_memory=False)
                    sub["format"] = name
                    dfs.append(sub)
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded raw Hawkeye CSVs: {len(combined):,} rows")
            return combined

        logger.warning("No Hawkeye dataset files found. Returning empty DataFrame.")
        return pd.DataFrame()

    def prepare_no_ball_flags(self) -> pd.DataFrame:
        """
        Ensure is_no_ball and standard columns are properly formed.
        """
        if self.df.empty:
            return self.df

        df = self.df.copy()

        # Parse is_no_ball flag if not present
        if "is_no_ball" not in df.columns:
            if "extras" in df.columns:
                df["is_no_ball"] = df["extras"].astype(str).str.contains("Nb", na=False).astype(int)
            elif "extra_nb" in df.columns:
                df["is_no_ball"] = (df["extra_nb"] > 0).astype(int)
            else:
                df["is_no_ball"] = 0

        # Ensure bowler identifier
        if "bowler" not in df.columns and "bowler_id" in df.columns:
            df["bowler"] = df["bowler_id"]
        elif "bowler" not in df.columns:
            df["bowler"] = "unknown_bowler"

        return df

    def compute_overall_stats(self) -> Dict[str, Any]:
        """Compute overall no-ball rate and dataset-level counts."""
        df = self.prepare_no_ball_flags()
        if df.empty:
            return {"error": "No data available"}

        total_deliveries = len(df)
        total_no_balls = int(df["is_no_ball"].sum())
        no_ball_rate_pct = (total_no_balls / total_deliveries) * 100.0 if total_deliveries > 0 else 0.0

        return {
            "total_deliveries": total_deliveries,
            "total_no_balls": total_no_balls,
            "overall_no_ball_rate_pct": round(no_ball_rate_pct, 4),
            "deliveries_per_no_ball": round(total_deliveries / total_no_balls, 1) if total_no_balls > 0 else None,
        }

    def analyze_bowler_no_ball_rates(
        self, min_deliveries: int = 120
    ) -> pd.DataFrame:
        """
        Compute bowler-level no-ball rates and identify highest frequency bowlers.

        Args:
            min_deliveries: Minimum deliveries bowled to be included (default 120 = 20 overs).

        Returns:
            DataFrame sorted by no_ball_rate descending.
        """
        df = self.prepare_no_ball_flags()
        if df.empty or "bowler" not in df.columns:
            return pd.DataFrame()

        agg_dict = {
            "deliveries": ("is_no_ball", "count"),
            "no_balls": ("is_no_ball", "sum"),
            "no_ball_rate_pct": ("is_no_ball", lambda x: (x.sum() / len(x)) * 100.0),
        }

        if "bowling_style" in df.columns:
            agg_dict["bowling_style"] = ("bowling_style", lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")

        if "ball_speed_kmh" in df.columns:
            agg_dict["avg_speed_kmh"] = ("ball_speed_kmh", "mean")

        grouped = df.groupby("bowler").agg(**agg_dict).reset_index()
        filtered = grouped[grouped["deliveries"] >= min_deliveries].copy()
        filtered["no_ball_rate_pct"] = filtered["no_ball_rate_pct"].round(3)

        if "avg_speed_kmh" in filtered.columns:
            filtered["avg_speed_kmh"] = filtered["avg_speed_kmh"].round(1)

        return filtered.sort_values(by="no_ball_rate_pct", ascending=False)

    def analyze_by_format(self) -> pd.DataFrame:
        """Compute no-ball statistics broken down by match format."""
        df = self.prepare_no_ball_flags()
        if df.empty or "format" not in df.columns:
            return pd.DataFrame()

        fmt_stats = df.groupby("format").agg(
            total_deliveries=("is_no_ball", "count"),
            no_balls=("is_no_ball", "sum"),
            no_ball_rate_pct=("is_no_ball", lambda x: (x.sum() / len(x)) * 100.0),
        ).reset_index()

        fmt_stats["no_ball_rate_pct"] = fmt_stats["no_ball_rate_pct"].round(4)
        return fmt_stats.sort_values(by="no_ball_rate_pct", ascending=False)

    def analyze_by_bowling_style(self) -> pd.DataFrame:
        """Compute no-ball frequency across seamers vs spinners."""
        df = self.prepare_no_ball_flags()
        if df.empty or "bowling_style" not in df.columns:
            return pd.DataFrame()

        style_stats = df.groupby("bowling_style").agg(
            total_deliveries=("is_no_ball", "count"),
            no_balls=("is_no_ball", "sum"),
            no_ball_rate_pct=("is_no_ball", lambda x: (x.sum() / len(x)) * 100.0),
        ).reset_index()

        style_stats["no_ball_rate_pct"] = style_stats["no_ball_rate_pct"].round(4)
        return style_stats[style_stats["total_deliveries"] >= 100].sort_values(
            by="no_ball_rate_pct", ascending=False
        )

    def analyze_by_match_phase(self) -> pd.DataFrame:
        """Analyze no-ball frequency across innings phases (over buckets)."""
        df = self.prepare_no_ball_flags()
        if df.empty:
            return pd.DataFrame()

        if "ball_phase" in df.columns:
            phase_col = "ball_phase"
        elif "over_num" in df.columns:
            df["over_bucket"] = pd.cut(
                df["over_num"],
                bins=[-1, 5, 15, 20, 50, 100],
                labels=["0-5 (Powerplay)", "6-15 (Middle)", "16-20 (Death T20)", "21-50 (ODI Middle/Late)", "50+ (Test)"]
            )
            phase_col = "over_bucket"
        else:
            return pd.DataFrame()

        phase_stats = df.groupby(phase_col, observed=True).agg(
            total_deliveries=("is_no_ball", "count"),
            no_balls=("is_no_ball", "sum"),
            no_ball_rate_pct=("is_no_ball", lambda x: (x.sum() / len(x)) * 100.0),
        ).reset_index()

        phase_stats["no_ball_rate_pct"] = phase_stats["no_ball_rate_pct"].round(4)
        return phase_stats

    def generate_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive dictionary report of no-ball patterns."""
        overall = self.compute_overall_stats()
        format_df = self.analyze_by_format()
        style_df = self.analyze_by_bowling_style()
        phase_df = self.analyze_by_match_phase()
        top_bowlers_df = self.analyze_bowler_no_ball_rates(min_deliveries=120)

        report = {
            "overall_statistics": overall,
            "format_breakdown": format_df.to_dict(orient="records") if not format_df.empty else [],
            "bowling_style_breakdown": style_df.to_dict(orient="records") if not style_df.empty else [],
            "match_phase_breakdown": phase_df.to_dict(orient="records") if not phase_df.empty else [],
            "top_no_ball_bowlers": (
                top_bowlers_df.head(10).to_dict(orient="records") if not top_bowlers_df.empty else []
            ),
        }
        return report


# ────────────────────────────────────────────────────────────────────────────
# 5. Training, Evaluation & Reporting Runner
# ────────────────────────────────────────────────────────────────────────────

def run_no_ball_training_and_eval(
    dataset_dir: Optional[Union[str, Path]] = None,
    save_model_bundle: bool = True
) -> Dict[str, Any]:
    """
    Load dataset, train HOG+SVM model, evaluate via LOOCV, and optionally save model.

    Returns:
        Dictionary of LOOCV evaluation metrics.
    """
    print("=" * 70)
    print("NO-BALL DETECTION MODULE: Image Classification Pipeline (HOG + SVM)")
    print("=" * 70)

    loader = NoBallDatasetLoader(dataset_dir)
    file_paths, labels, class_names = loader.load_dataset()

    if len(file_paths) == 0:
        print("[WARNING] No images found. Check dataset path:", loader.dataset_dir)
        return {"error": "no_images_found"}

    classifier = NoBallVisionClassifier()
    X, y, valid_paths = classifier.extract_features_matrix(file_paths, labels)

    print(f"Extracted HOG features: {X.shape[0]} samples, {X.shape[1]} features per image.")

    # Run Leave-One-Out Cross-Validation
    results = classifier.evaluate_leave_one_out(X, y)

    print("\n--- LOOCV Evaluation Metrics ---")
    print(f"  Total Samples:    {results['n_samples']} ({results['n_legal']} legal, {results['n_no_ball']} no-ball)")
    print(f"  Accuracy:         {results['accuracy'] * 100:.2f}%")
    print(f"  Precision:        {results['precision'] * 100:.2f}%")
    print(f"  Recall (Sens.):   {results['recall'] * 100:.2f}%")
    print(f"  Specificity:      {results['specificity'] * 100:.2f}%")
    print(f"  F1 Score:         {results['f1_score']:.4f}")
    print(f"  ROC-AUC:          {results['roc_auc']:.4f}")
    print(f"  Confusion Matrix: {results['confusion_matrix']}")

    # Fit final pipeline on all data
    classifier.fit(X, y)

    if save_model_bundle:
        saved_path = classifier.save()
        print(f"\n[OK] Model successfully saved to: {saved_path}")

        # Save evaluation JSON
        eval_json_path = EXPERIMENTS / "results" / "no_ball_eval.json"
        with open(eval_json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Evaluation results saved to: {eval_json_path}")

    return results


def run_hawkeye_statistical_analysis(
    output_json_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run HawkeyeStats statistical patterns for extras no-ball column.
    """
    print("\n" + "=" * 70)
    print("HAWKEYESTATS NO-BALL PATTERN ANALYSIS")
    print("=" * 70)

    analyzer = HawkeyeNoBallAnalysis()
    report = analyzer.generate_full_report()

    overall = report.get("overall_statistics", {})
    print(f"  Total Deliveries Analyzed: {overall.get('total_deliveries', 0):,}")
    print(f"  Total No-Balls Recorded:   {overall.get('total_no_balls', 0):,}")
    print(f"  Overall No-Ball Rate:      {overall.get('overall_no_ball_rate_pct', 0.0):.4f}%")
    print(f"  Deliveries Per No-Ball:    1 every {overall.get('deliveries_per_no_ball', 'N/A')} balls")

    print("\n--- No-Ball Rate by Match Format ---")
    for row in report.get("format_breakdown", []):
        print(f"  {row['format']:<12}: {row['no_ball_rate_pct']:.4f}% ({row['no_balls']}/{row['total_deliveries']})")

    print("\n--- Top Bowlers by No-Ball Frequency (min 120 balls) ---")
    for i, row in enumerate(report.get("top_no_ball_bowlers", [])[:5], start=1):
        style = row.get('bowling_style', 'N/A')
        print(f"  {i}. {row['bowler']:<20} ({style:<12}): {row['no_ball_rate_pct']:.2f}% "
              f"({row['no_balls']} NBs in {row['deliveries']} balls)")

    out_file = output_json_path or (EXPERIMENTS / "results" / "hawkeye_no_ball_stats.json")
    try:
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[OK] Statistical analysis report saved to: {out_file}")
    except Exception as e:
        logger.error(f"Failed to write stats JSON: {e}")

    return report


# ────────────────────────────────────────────────────────────────────────────
# 6. Main Entry Point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Train and evaluate HOG+SVM vision model on vibhudave dataset
    vision_metrics = run_no_ball_training_and_eval()

    # 2. Run statistical analysis across Hawkeye deliveries
    stats_report = run_hawkeye_statistical_analysis()

    print("\n" + "=" * 70)
    print("MODULE 12 EXECUTION COMPLETE")
    print("=" * 70)
