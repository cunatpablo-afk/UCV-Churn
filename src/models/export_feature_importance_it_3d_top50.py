from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.train_models as base
from src.utils.paths import MODEL_REPORTS_DIR, MODELING_DIR, MODELS_DIR, ensure_dir

RUN_SUFFIX = "it_3d_top50"
DATASET_PATH = MODELING_DIR / "churn_modeling_dataset_it_3d_top50.csv"
MODEL_PATH = MODELS_DIR / f"best_model_{RUN_SUFFIX}.joblib"
TOP_N = 30


def extract_original_feature(feature_name: str, original_columns: list[str]) -> str:
    cleaned = feature_name.replace("num__", "").replace("cat__", "")
    cleaned = cleaned.replace("missingindicator_", "")

    if cleaned in original_columns:
        return cleaned

    matches = [col for col in original_columns if cleaned.startswith(f"{col}_")]
    if matches:
        return max(matches, key=len)

    return cleaned


def get_model_importance(model) -> tuple[np.ndarray, str]:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        return estimator.coef_[0], "coefficient"
    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_, "feature_importance"
    raise TypeError(f"El modelo {type(estimator).__name__} no expone importancias.")


def export_feature_importance() -> dict[str, Path]:
    ensure_dir(MODEL_REPORTS_DIR)
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe {DATASET_PATH}. Ejecuta primero build_reduced_dataset_it_3d_top50.py.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe {MODEL_PATH}. Ejecuta primero train_models_it_3d_top50.py.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATASET_PATH, parse_dates=["fecha"])
    train_df, _ = base.temporal_split(df)
    X_train, _ = base.get_feature_matrix(train_df)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importance_values, importance_type = get_model_importance(model)
    if len(feature_names) != len(importance_values):
        raise ValueError(f"No coinciden features e importancias: {len(feature_names)} vs {len(importance_values)}")

    importance = pd.DataFrame(
        {
            "feature_transformed": feature_names,
            "importance": importance_values,
            "abs_importance": np.abs(importance_values),
            "importance_type": importance_type,
        }
    )
    importance["feature_original"] = importance["feature_transformed"].map(
        lambda value: extract_original_feature(value, X_train.columns.tolist())
    )
    importance = importance.sort_values("abs_importance", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)

    grouped = (
        importance.groupby("feature_original", as_index=False)
        .agg(
            abs_importance=("abs_importance", "sum"),
            max_abs_importance=("abs_importance", "max"),
            n_transformed_features=("feature_transformed", "count"),
        )
        .sort_values("abs_importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["rank"] = np.arange(1, len(grouped) + 1)

    importance_path = MODEL_REPORTS_DIR / f"feature_importance_{RUN_SUFFIX}.csv"
    grouped_path = MODEL_REPORTS_DIR / f"feature_importance_grouped_{RUN_SUFFIX}.csv"
    top30_path = MODEL_REPORTS_DIR / f"top_features_{TOP_N}_{RUN_SUFFIX}.csv"

    importance.to_csv(importance_path, index=False)
    grouped.to_csv(grouped_path, index=False)
    grouped.head(TOP_N).to_csv(top30_path, index=False)

    return {
        "feature_importance": importance_path,
        "feature_importance_grouped": grouped_path,
        "top_features_30": top30_path,
    }


if __name__ == "__main__":
    for name, path in export_feature_importance().items():
        print(f"{name}: {path}")
