from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import MODELING_DIR, ensure_dir, project_path

## Iteracion 7:
## Partimos de 3C Top75 y anadimos variables manuales interpretables.
## Todas se calculan con informacion disponible en t o antes:
## - caida_facturacion_3m: compara facturacion actual contra t-3.
## - ratio_facturacion_3m: relacion facturacion actual / facturacion t-3.
## - deterioro_calidad_3m: compara calidad actual contra t-3.
## - tickets_ultimos_90d: suma contactos soporte en t, t-1 y t-2.
##
## Nota de sesgo:
## En la siguiente iteracion conviene auditar y excluir genero si aparece en el
## dataset usado, para evitar que el modelo aprenda patrones sensibles.

BASE_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3c_top75.csv"
SOURCE_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3a.csv"
OUTPUT_DATASET = MODELING_DIR / "churn_modeling_dataset_it_7_manual_features.csv"
REPORT_CSV = MODELING_DIR / "manual_features_report_it_7.csv"
REPORT_JSON = MODELING_DIR / "manual_features_summary_it_7.json"

ID_COLS = ["cliente_id", "fecha"]
TARGET = "churn_t_plus_1"
MANUAL_FEATURES = [
    "caida_facturacion_3m",
    "ratio_facturacion_3m",
    "deterioro_calidad_3m",
    "tickets_ultimos_90d",
]
REQUIRED_SOURCE_COLS = [
    "cliente_id",
    "fecha",
    "fact_importe_total",
    "fact_importe_total_lag_3m",
    "red_indice_calidad_global",
    "red_indice_calidad_global_lag_3m",
    "soporte_contactos",
    "soporte_contactos_lag_1m",
    "soporte_contactos_lag_2m",
]


def _validate_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {dataset_name}: {missing}")


def _build_manual_features(source: pd.DataFrame) -> pd.DataFrame:
    features = source[REQUIRED_SOURCE_COLS].copy()

    features["caida_facturacion_3m"] = features["fact_importe_total_lag_3m"] - features["fact_importe_total"]
    features["ratio_facturacion_3m"] = np.where(
        features["fact_importe_total_lag_3m"].abs() > 0,
        features["fact_importe_total"] / features["fact_importe_total_lag_3m"],
        np.nan,
    )
    features["deterioro_calidad_3m"] = (
        features["red_indice_calidad_global_lag_3m"] - features["red_indice_calidad_global"]
    )
    features["tickets_ultimos_90d"] = (
        features["soporte_contactos"].fillna(0)
        + features["soporte_contactos_lag_1m"].fillna(0)
        + features["soporte_contactos_lag_2m"].fillna(0)
    )

    return features[ID_COLS + MANUAL_FEATURES]


def build_dataset() -> dict[str, Path]:
    ensure_dir(MODELING_DIR)
    if not BASE_DATASET.exists():
        raise FileNotFoundError(f"No existe {BASE_DATASET}. Ejecuta primero la Iteracion 3C.")
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"No existe {SOURCE_DATASET}. Ejecuta primero la Iteracion 3A.")

    base = pd.read_csv(BASE_DATASET, parse_dates=["fecha"])
    source = pd.read_csv(SOURCE_DATASET, parse_dates=["fecha"], usecols=REQUIRED_SOURCE_COLS)

    _validate_columns(base, ID_COLS + [TARGET], BASE_DATASET.name)
    _validate_columns(source, REQUIRED_SOURCE_COLS, SOURCE_DATASET.name)

    manual = _build_manual_features(source)
    output = base.merge(manual, on=ID_COLS, how="left", validate="one_to_one")

    duplicated = int(output.duplicated(ID_COLS).sum())
    if duplicated:
        raise ValueError(f"Dataset it_7 contiene duplicados cliente-mes: {duplicated}")

    output.to_csv(OUTPUT_DATASET, index=False)

    rows = []
    for feature in MANUAL_FEATURES:
        rows.append(
            {
                "feature": feature,
                "missing_count": int(output[feature].isna().sum()),
                "missing_rate": float(output[feature].isna().mean()),
                "mean": float(output[feature].mean(skipna=True)),
                "std": float(output[feature].std(skipna=True)),
                "min": float(output[feature].min(skipna=True)),
                "max": float(output[feature].max(skipna=True)),
            }
        )
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False)

    summary = {
        "iteration": "it_7_manual_features",
        "base_dataset": project_path(BASE_DATASET),
        "source_dataset": project_path(SOURCE_DATASET),
        "output_dataset": project_path(OUTPUT_DATASET),
        "leakage_control": "Todas las variables usan informacion de t, t-1, t-2 o t-3; no usan churn futuro.",
        "manual_features": {
            "caida_facturacion_3m": "fact_importe_total_lag_3m - fact_importe_total",
            "ratio_facturacion_3m": "fact_importe_total / fact_importe_total_lag_3m, con denominador 0 como NaN",
            "deterioro_calidad_3m": "red_indice_calidad_global_lag_3m - red_indice_calidad_global",
            "tickets_ultimos_90d": "soporte_contactos + soporte_contactos_lag_1m + soporte_contactos_lag_2m",
        },
        "bias_note_next_iteration": (
            "Auditar genero y excluirlo si aparece en el dataset de entrenamiento para reducir riesgo de sesgo."
        ),
        "rows": int(len(output)),
        "base_columns": int(base.shape[1]),
        "output_columns": int(output.shape[1]),
        "new_feature_columns": MANUAL_FEATURES,
        "genero_present_in_base": bool("genero" in base.columns),
        "genero_present_in_output": bool("genero" in output.columns),
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "dataset": OUTPUT_DATASET,
        "report_csv": REPORT_CSV,
        "report_json": REPORT_JSON,
    }


if __name__ == "__main__":
    for name, path in build_dataset().items():
        print(f"{name}: {path}")
