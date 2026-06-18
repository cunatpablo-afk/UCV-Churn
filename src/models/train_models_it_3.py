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
from src.utils.paths import MODELING_DIR

## Iteracion 3A:
## Dataset reducido a dominios de facturacion, soporte y calidad de red.
## Sin variables geograficas directas.
##
## Recordatorio para interpretar resultados:
## - 3A: sin variables geograficas.
## - 3B pendiente: con geografia agregada no identificativa, sin IDs directos.
## - Comparacion principal: mismos hiperparametros de Iteracion 2.
## - Comparacion secundaria opcional: pequeno tuning solo de XGBoost reducido.
## - La seleccion por nombre es practica pero imperfecta: revisar el reporte de columnas.

REDUCED_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3a.csv"

base.RUN_SUFFIX = "it_3a"
base.SEARCH_MODE = "none"
base.N_ITER = 5
base.TRAIN_SAMPLE_FRAC = 1.0


def load_modeling_dataset() -> pd.DataFrame:
    base.log(f"Cargando dataset reducido de modelado: {REDUCED_DATASET}")
    if not REDUCED_DATASET.exists():
        raise FileNotFoundError(
            f"No existe {REDUCED_DATASET}. Ejecuta primero src/features/build_reduced_dataset_it_3a.py."
        )
    df = pd.read_csv(REDUCED_DATASET, parse_dates=["fecha"])
    if base.TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo {base.TARGET}")
    base.log(f"Dataset reducido cargado: {len(df):,} filas, {df.shape[1]:,} columnas")
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


base.load_modeling_dataset = load_modeling_dataset
base.get_models = get_models


if __name__ == "__main__":
    for name, path in base.run_training().items():
        print(f"{name}: {path}")
