from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_reduced_dataset_it_3a import ID_AND_TARGET_COLS, classify_column
from src.utils.paths import MODELING_DIR, ensure_dir, project_path

INPUT_DATASET = MODELING_DIR / "churn_modeling_dataset.csv"
OUTPUT_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3b.csv"
REPORT_CSV = MODELING_DIR / "feature_selection_report_it_3b.csv"
REPORT_JSON = MODELING_DIR / "feature_selection_summary_it_3b.json"

## Iteracion 3B:
## Partimos de 3A y anadimos contexto geografico agregado.
## No exportamos identificadores directos como zona_id, region o tipo_zona.
## Tampoco calculamos ningun agregado del target/churn para evitar leakage.

ZONE_GROUP = ["fecha", "zona_id"]
REGION_GROUP = ["fecha", "region"]

AGG_SOURCE_COLUMNS = [
    "fact_importe_total",
    "fact_impago_flag",
    "fact_dias_retraso_pago",
    "fact_consumo_extra",
    "fact_variacion_consumo_pct",
    "soporte_contactos",
    "soporte_no_resueltos",
    "soporte_satisfaccion_media",
    "soporte_resuelto_rate",
    "soporte_impago_mes",
    "red_indice_calidad_global",
    "red_latencia_ms",
    "red_tasa_cortes_pct",
    "red_velocidad_media_mbps",
    "red_cobertura_5g_pct",
]

RELATIVE_COLUMNS = [
    "fact_importe_total",
    "fact_dias_retraso_pago",
    "fact_consumo_extra",
    "soporte_contactos",
    "soporte_no_resueltos",
    "soporte_satisfaccion_media",
    "red_indice_calidad_global",
    "red_latencia_ms",
    "red_tasa_cortes_pct",
]


def add_group_context(df: pd.DataFrame, group_cols: list[str], prefix: str) -> list[str]:
    created_columns = []
    for col in AGG_SOURCE_COLUMNS:
        if col not in df.columns:
            continue

        new_col = f"{prefix}_{col}_mean"
        df[new_col] = df.groupby(group_cols, dropna=False)[col].transform("mean")
        created_columns.append(new_col)

    return created_columns


def add_relative_context(df: pd.DataFrame, prefix: str) -> list[str]:
    created_columns = []
    for col in RELATIVE_COLUMNS:
        context_col = f"{prefix}_{col}_mean"
        if col not in df.columns or context_col not in df.columns:
            continue

        new_col = f"rel_{col}_vs_{prefix.replace('geo_', '')}"
        df[new_col] = df[col] - df[context_col]
        created_columns.append(new_col)

    return created_columns


def build_reduced_dataset() -> dict[str, Path]:
    ensure_dir(MODELING_DIR)
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"No existe {INPUT_DATASET}. Ejecuta primero feature engineering.")

    df = pd.read_csv(INPUT_DATASET, parse_dates=["fecha"])

    selected_columns = []
    report_rows = []
    for column in df.columns:
        selected, domain, reason = classify_column(column)
        report_rows.append(
            {
                "column": column,
                "selected": int(selected),
                "domain": domain,
                "reason": reason,
            }
        )
        if selected:
            selected_columns.append(column)

    geo_columns = []
    geo_columns.extend(add_group_context(df, ZONE_GROUP, "geo_zona"))
    geo_columns.extend(add_group_context(df, REGION_GROUP, "geo_region"))
    geo_columns.extend(add_relative_context(df, "geo_zona"))
    geo_columns.extend(add_relative_context(df, "geo_region"))

    for column in geo_columns:
        report_rows.append(
            {
                "column": column,
                "selected": 1,
                "domain": "geo_agregada",
                "reason": "contexto geografico agregado no identificativo",
            }
        )

    output_columns = selected_columns + geo_columns
    direct_geo_leaks = [c for c in output_columns if c in {"zona_id", "fact_zona_id", "region", "red_region", "tipo_zona", "red_tipo_zona"}]
    if direct_geo_leaks:
        raise ValueError(f"Columnas geograficas directas detectadas en salida: {direct_geo_leaks}")

    target_leaks = [c for c in output_columns if c.lower() in {"churn", "ever_churn"}]
    if target_leaks:
        raise ValueError(f"Columnas prohibidas detectadas en salida: {target_leaks}")

    reduced = df[output_columns].copy()
    reduced.to_csv(OUTPUT_DATASET, index=False)

    report = pd.DataFrame(report_rows)
    report.to_csv(REPORT_CSV, index=False)

    selected_features = [c for c in output_columns if c not in ID_AND_TARGET_COLS]
    removed_columns = [c for c in df.columns if c not in output_columns and c not in geo_columns]
    summary = {
        "iteration": "it_3b",
        "input_dataset": project_path(INPUT_DATASET),
        "output_dataset": project_path(OUTPUT_DATASET),
        "selection_rule": "it_3a + agregados por zona/region sin identificadores directos",
        "important_notes_for_results_review": [
            "Iteracion 3A: sin variables geograficas directas.",
            "Iteracion 3B: geografia agregada no identificativa, sin IDs directos.",
            "La seleccion por nombre es practica pero imperfecta; revisar manualmente columnas conservadas/eliminadas.",
            "Los agregados no usan churn ni churn_t_plus_1.",
            "Comparacion principal: mismos hiperparametros de Iteracion 2.",
            "Comparacion secundaria opcional: pequeno tuning solo de XGBoost reducido si el rendimiento cae o queda cerca.",
        ],
        "rows": int(len(reduced)),
        "original_columns_total": int(len(pd.read_csv(INPUT_DATASET, nrows=1).columns)),
        "selected_columns_total": int(len(output_columns)),
        "selected_feature_columns": int(len(selected_features)),
        "geo_aggregate_columns": geo_columns,
        "geo_aggregate_columns_total": int(len(geo_columns)),
        "removed_columns": [c for c in pd.read_csv(INPUT_DATASET, nrows=1).columns if c not in output_columns],
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "dataset": OUTPUT_DATASET,
        "report_csv": REPORT_CSV,
        "report_json": REPORT_JSON,
    }


if __name__ == "__main__":
    for name, path in build_reduced_dataset().items():
        print(f"{name}: {path}")
