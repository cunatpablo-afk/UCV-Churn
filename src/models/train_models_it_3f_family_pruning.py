from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.train_models as base
from src.models.train_models_it_3e_top30 import (
    export_feature_importance,
    export_multicollinearity_report,
    export_overfitting_report,
)
from src.utils.paths import MODEL_REPORTS_DIR, MODELING_DIR

## Iteracion 3F:
## Top50 con poda por familias correlacionadas y eliminacion de proxies
## geograficos red_poblacion_zona_*.
## Sin tuning: mismos hiperparametros que 3D para aislar el efecto de la poda.

RUN_SUFFIX = "it_3f_family_pruning"
DATASET = MODELING_DIR / "churn_modeling_dataset_it_3f_family_pruning.csv"

base.RUN_SUFFIX = RUN_SUFFIX
base.SEARCH_MODE = "none"
base.N_ITER = 5
base.TRAIN_SAMPLE_FRAC = 1.0


def load_modeling_dataset() -> pd.DataFrame:
    base.log(f"Cargando dataset 3F de modelado: {DATASET}")
    if not DATASET.exists():
        raise FileNotFoundError(
            f"No existe {DATASET}. Ejecuta primero src/features/build_reduced_dataset_it_3f_family_pruning.py."
        )
    df = pd.read_csv(DATASET, parse_dates=["fecha"])
    if base.TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo {base.TARGET}")
    base.log(f"Dataset 3F cargado: {len(df):,} filas, {df.shape[1]:,} columnas")
    return df


def get_models(y_train: pd.Series):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    return {
        "dummy_mayoritaria": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            C=0.1,
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=base.SEED,
            n_jobs=base.N_JOBS_MODEL,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=3,
            reg_alpha=1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=base.SEED,
            n_jobs=base.N_JOBS_MODEL,
        ),
    }


def export_iteration_comparison() -> Path:
    rows = []
    configs = [
        ("3C Top75", "it_3c_top75", 75),
        ("3D Top50", "it_3d_top50", 50),
        ("3E Top30", "it_3e_top30", 30),
        ("3F Family pruning", RUN_SUFFIX, 37),
    ]
    diagnostics_path = MODEL_REPORTS_DIR / f"overfitting_diagnostics_{RUN_SUFFIX}.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    for label, suffix, n_features in configs:
        ranking_path = MODEL_REPORTS_DIR / f"model_ranking_{suffix}.csv"
        metrics_path = MODEL_REPORTS_DIR / f"model_metrics_{suffix}.csv"
        if not ranking_path.exists():
            continue
        ranking = pd.read_csv(ranking_path).sort_values("rank")
        best = ranking.iloc[0]
        model_name = best["model"]
        precision = pd.NA
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            match_metrics = metrics[(metrics["model"] == model_name) & (metrics["split"] == "test_temporal")]
            if not match_metrics.empty:
                precision = match_metrics.iloc[0]["precision"]
        diagnostic = ""
        if suffix == RUN_SUFFIX and not diagnostics.empty:
            match = diagnostics[diagnostics["model"] == model_name]
            if not match.empty:
                diagnostic = str(match.iloc[0]["diagnostico"])
        rows.append(
            {
                "iteration": label,
                "suffix": suffix,
                "best_model": model_name,
                "n_features": n_features,
                "pr_auc_test": best["pr_auc_test"],
                "roc_auc_test": best["roc_auc_test"],
                "recall_test": best["recall_test"],
                "precision_test": precision,
                "f1_test": best["f1_test"],
                "feature_reduction_vs_3c": 75 - n_features,
                "overfitting_diagnostic": diagnostic,
            }
        )
    output_path = MODEL_REPORTS_DIR / f"iteration_comparison_{RUN_SUFFIX}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


base.load_modeling_dataset = load_modeling_dataset
base.get_models = get_models

## Reutilizamos los exportadores de 3E, cambiando sus globals para que escriban
## con sufijo 3F y lean el dataset 3F.
import src.models.train_models_it_3e_top30 as diagnostics

diagnostics.RUN_SUFFIX = RUN_SUFFIX
diagnostics.DATASET = DATASET


if __name__ == "__main__":
    outputs = base.run_training()
    outputs["overfitting"] = export_overfitting_report()
    outputs.update(export_feature_importance())
    outputs.update(export_multicollinearity_report())
    outputs["comparison"] = export_iteration_comparison()
    for name, path in outputs.items():
        print(f"{name}: {path}")
