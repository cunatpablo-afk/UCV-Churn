from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.train_models as base
from src.models.train_models_it_3e_top30 import (
    export_feature_importance,
    export_multicollinearity_report,
    export_overfitting_report,
)
from src.utils.paths import MODELING_DIR

## Iteracion 3G:
## Prueba especifica para Logistic Regression con imputacion mas semantica.
## Base: dataset 3F, porque ya elimina redundancias fuertes y proxies geograficos.
##
## Reglas:
## - Numericas de eventos/conteos/flags: imputacion a 0 + indicador de nulo.
## - Resto de numericas: mediana + indicador de nulo.
## - Categoricas: "Desconocido" + one-hot.
##
## Objetivo: comprobar si una imputacion mas fiel al negocio mejora o mantiene
## Logistic frente a la imputacion generica de mediana/moda.

RUN_SUFFIX = "it_3g_logistic_semantic_imputation"
DATASET = MODELING_DIR / "churn_modeling_dataset_it_3f_family_pruning.csv"

base.RUN_SUFFIX = RUN_SUFFIX
base.SEARCH_MODE = "none"
base.N_ITER = 5
base.TRAIN_SAMPLE_FRAC = 1.0


def load_modeling_dataset() -> pd.DataFrame:
    base.log(f"Cargando dataset 3G de modelado: {DATASET}")
    if not DATASET.exists():
        raise FileNotFoundError(
            f"No existe {DATASET}. Ejecuta primero src/features/build_reduced_dataset_it_3f_family_pruning.py."
        )
    df = pd.read_csv(DATASET, parse_dates=["fecha"])
    if base.TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo {base.TARGET}")
    base.log(f"Dataset 3G cargado: {len(df):,} filas, {df.shape[1]:,} columnas")
    return df


def _uses_zero_imputation(col: str) -> bool:
    name = col.lower()
    if name.endswith("_was_missing") or "missing_rate" in name:
        return True
    if name.startswith("soporte_") and any(
        term in name
        for term in [
            "contactos",
            "no_resueltos",
            "incidencia",
            "impago",
            "duracion",
        ]
    ):
        return True
    if name.startswith("fact_impago_flag") or name.startswith("fact_incidencia_masiva"):
        return True
    if name.startswith("red_incidencia_masiva"):
        return True
    if name.startswith("encuestas_n"):
        return True
    return False


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    zero_numeric_cols = [col for col in numeric_cols if _uses_zero_imputation(col)]
    median_numeric_cols = [col for col in numeric_cols if col not in zero_numeric_cols]

    zero_numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    median_numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if zero_numeric_cols:
        transformers.append(("num_zero", zero_numeric_transformer, zero_numeric_cols))
    if median_numeric_cols:
        transformers.append(("num_median", median_numeric_transformer, median_numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    base.log(
        "Preprocesado semantico: "
        f"numericas_zero={len(zero_numeric_cols)}, "
        f"numericas_mediana={len(median_numeric_cols)}, "
        f"categoricas_desconocido={len(categorical_cols)}"
    )
    return ColumnTransformer(transformers)


def get_models(y_train: pd.Series):
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
    }


base.load_modeling_dataset = load_modeling_dataset
base.build_preprocessor = build_preprocessor
base.get_models = get_models

## Reutilizamos los exportadores de diagnostico, cambiando sus globals para que
## escriban con sufijo 3G y lean el dataset 3F.
import src.models.train_models_it_3e_top30 as diagnostics

diagnostics.RUN_SUFFIX = RUN_SUFFIX
diagnostics.DATASET = DATASET


if __name__ == "__main__":
    outputs = base.run_training()
    outputs["overfitting"] = export_overfitting_report()
    outputs.update(export_feature_importance())
    outputs.update(export_multicollinearity_report())
    for name, path in outputs.items():
        print(f"{name}: {path}")
