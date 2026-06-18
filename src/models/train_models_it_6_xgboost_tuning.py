from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.train_models as base
from src.utils.paths import MODEL_REPORTS_DIR, MODELING_DIR, project_path

## Iteracion 6:
## Tuning avanzado de XGBoost sobre el dataset simplificado 3C Top75.
##
## Objetivo:
## - Mantener Logistic Regression como baseline fuerte y estable.
## - Ajustar XGBoost con una busqueda mas amplia.
## - Comparar despues no solo PR-AUC, tambien Lift@5/10 y falsos positivos.
##
## Importante:
## Esta iteracion cambia hiperparametros, no variables. La comparacion justa es
## contra Iteracion 3C Top75, que usa el mismo dataset con hiperparametros fijos.

DATASET = MODELING_DIR / "churn_modeling_dataset_it_3c_top75.csv"
RUNTIME_REPORT = MODEL_REPORTS_DIR / "runtime_it_6_xgb_tuned.json"
XGBOOST_DEVICE = "cuda"

base.RUN_SUFFIX = "it_6_xgb_tuned"
base.SEARCH_MODE = "randomized"
base.N_ITER = 30
base.CV_SPLITS = 3
base.N_JOBS_MODEL = 4
base.N_JOBS_SEARCH = 1
base.TRAIN_SAMPLE_FRAC = 1.0

base.PARAM_GRIDS = {
    "xgboost": {
        "model__n_estimators": [80, 100, 150, 200, 300],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "model__subsample": [0.65, 0.75, 0.85, 1.0],
        "model__colsample_bytree": [0.65, 0.75, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5, 8, 12],
        "model__gamma": [0, 0.5, 1, 2],
        "model__reg_alpha": [0, 0.25, 0.5, 1, 2, 5],
        "model__reg_lambda": [1, 2, 3, 5, 8, 12],
    }
}


def load_modeling_dataset() -> pd.DataFrame:
    base.log(f"Cargando dataset de Iteracion 6: {DATASET}")
    if not DATASET.exists():
        raise FileNotFoundError(f"No existe {DATASET}. Ejecuta primero Iteracion 3C Top75.")
    df = pd.read_csv(DATASET, parse_dates=["fecha"])
    if base.TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo {base.TARGET}")
    base.log(f"Dataset it_6 cargado: {len(df):,} filas, {df.shape[1]:,} columnas")
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
            tree_method="hist",
            device=XGBOOST_DEVICE,
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


base.load_modeling_dataset = load_modeling_dataset
base.get_models = get_models


if __name__ == "__main__":
    start = time.perf_counter()
    base.log(f"Iteracion 6 usando XGBoost device={XGBOOST_DEVICE}")
    outputs = base.run_training()
    elapsed_seconds = time.perf_counter() - start
    runtime = {
        "iteration": "it_6_xgb_tuned",
        "xgboost_device": XGBOOST_DEVICE,
        "total_runtime_seconds": elapsed_seconds,
        "total_runtime_minutes": elapsed_seconds / 60,
        "n_iter": base.N_ITER,
        "cv_splits": base.CV_SPLITS,
        "dataset": project_path(DATASET),
    }
    RUNTIME_REPORT.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    outputs["runtime"] = RUNTIME_REPORT
    for name, path in outputs.items():
        print(f"{name}: {path}")
