from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_models import RUN_SUFFIX, TARGET, get_feature_matrix, load_modeling_dataset, temporal_split, with_suffix
from src.utils.paths import MODEL_REPORTS_DIR, MODELS_DIR, ensure_dir


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
        values = estimator.coef_[0]
        return values, "coefficient"

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        return values, "feature_importance"

    raise TypeError(f"El modelo {type(estimator).__name__} no expone coef_ ni feature_importances_.")


def export_feature_importance() -> dict[str, Path]:
    ensure_dir(MODEL_REPORTS_DIR)

    model_path = MODELS_DIR / with_suffix("best_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe {model_path}. Ejecuta primero train_models.py.")

    model = joblib.load(model_path)
    df = load_modeling_dataset()
    train_df, _ = temporal_split(df)
    X_train, _ = get_feature_matrix(train_df)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importance_values, importance_type = get_model_importance(model)

    if len(feature_names) != len(importance_values):
        raise ValueError(
            "No coinciden features transformadas e importancias: "
            f"{len(feature_names)} vs {len(importance_values)}"
        )

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

    importance_path = MODEL_REPORTS_DIR / with_suffix("feature_importance.csv")
    grouped_path = MODEL_REPORTS_DIR / with_suffix("feature_importance_grouped.csv")
    top_features_path = MODEL_REPORTS_DIR / with_suffix("top_features_50.csv")

    importance.to_csv(importance_path, index=False)
    grouped.to_csv(grouped_path, index=False)
    grouped.head(50).to_csv(top_features_path, index=False)

    return {
        "feature_importance": importance_path,
        "feature_importance_grouped": grouped_path,
        "top_features_50": top_features_path,
    }


if __name__ == "__main__":
    for name, path in export_feature_importance().items():
        print(f"{name}: {path}")
