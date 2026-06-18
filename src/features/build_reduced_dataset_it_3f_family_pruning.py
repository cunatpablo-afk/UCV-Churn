from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import MODELING_DIR, ensure_dir, project_path

INPUT_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3d_top50.csv"
OUTPUT_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3f_family_pruning.csv"
REPORT_CSV = MODELING_DIR / "feature_selection_report_it_3f_family_pruning.csv"
REPORT_JSON = MODELING_DIR / "feature_selection_summary_it_3f_family_pruning.json"

ID_AND_TARGET_COLS = ["cliente_id", "fecha", "churn_t_plus_1"]

## Iteracion 3F:
## Partimos de Top50 y eliminamos redundancias por familias correlacionadas.
## Mantener una representante por familia reduce multicolinealidad sin hacer
## otra seleccion ciega por ranking Top-N.
KEEP_REPRESENTATIVES = {
    "fact_importe_total_roll_3m": "representante_facturacion_importe",
    "fact_cargo_base_lag_1m": "representante_cargo_base",
    "fact_num_lineas": "representante_num_lineas",
    "red_indice_calidad_global_roll_3m": "representante_calidad_red",
}

DROP_BY_FAMILY = {
    "fact_importe_total": "redundante_con_fact_importe_total_roll_3m",
    "fact_importe_total_lag_1m": "redundante_con_fact_importe_total_roll_3m",
    "fact_importe_total_lag_2m": "redundante_con_fact_importe_total_roll_3m",
    "fact_importe_total_lag_3m": "redundante_con_fact_importe_total_roll_3m",
    "fact_cargo_base_lag_2m": "redundante_con_fact_cargo_base_lag_1m",
    "fact_cargo_base_lag_3m": "redundante_con_fact_cargo_base_lag_1m",
    "fact_num_lineas_lag_1m": "redundante_con_fact_num_lineas",
    "fact_num_lineas_roll_3m": "redundante_con_fact_num_lineas",
    "red_indice_calidad_global_lag_1m": "redundante_con_red_indice_calidad_global_roll_3m",
    "red_indice_calidad_global_lag_2m": "redundante_con_red_indice_calidad_global_roll_3m",
    "red_indice_calidad_global_lag_3m": "redundante_con_red_indice_calidad_global_roll_3m",
    "red_poblacion_zona_lag_2m": "proxy_geografico_eliminado",
    "red_poblacion_zona_lag_3m": "proxy_geografico_eliminado",
}


def build_reduced_dataset() -> dict[str, Path]:
    ensure_dir(MODELING_DIR)
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"No existe {INPUT_DATASET}. Ejecuta primero 3D Top50.")

    df = pd.read_csv(INPUT_DATASET, parse_dates=["fecha"])
    missing_representatives = [col for col in KEEP_REPRESENTATIVES if col not in df.columns]
    if missing_representatives:
        raise ValueError(f"Representantes no encontradas en 3D Top50: {missing_representatives}")

    drop_existing = [col for col in DROP_BY_FAMILY if col in df.columns]
    selected_columns = [col for col in df.columns if col not in drop_existing]
    reduced = df[selected_columns].copy()
    reduced.to_csv(OUTPUT_DATASET, index=False)

    rows = []
    for col in df.columns:
        if col in ID_AND_TARGET_COLS:
            selected = True
            reason = "id_target"
        elif col in DROP_BY_FAMILY:
            selected = False
            reason = DROP_BY_FAMILY[col]
        elif col in KEEP_REPRESENTATIVES:
            selected = True
            reason = KEEP_REPRESENTATIVES[col]
        else:
            selected = True
            reason = "conservada_top50_no_redundante"
        rows.append({"column": col, "selected": int(selected), "reason": reason})
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False)

    model_features = [col for col in selected_columns if col not in ID_AND_TARGET_COLS]
    summary = {
        "iteration": "it_3f_family_pruning",
        "input_dataset": project_path(INPUT_DATASET),
        "output_dataset": project_path(OUTPUT_DATASET),
        "selection_rule": "Top50 con poda manual por familias altamente correlacionadas y eliminacion de proxies geograficos red_poblacion_zona_*.",
        "rows": int(len(reduced)),
        "input_columns_total": int(df.shape[1]),
        "output_columns_total": int(reduced.shape[1]),
        "input_feature_columns": int(df.shape[1] - len(ID_AND_TARGET_COLS)),
        "output_feature_columns": int(len(model_features)),
        "kept_representatives": KEEP_REPRESENTATIVES,
        "dropped_by_family": {col: reason for col, reason in DROP_BY_FAMILY.items() if col in df.columns},
        "model_feature_columns": model_features,
        "important_notes_for_results_review": [
            "cliente_id, fecha y churn_t_plus_1 se conservan en el CSV, pero no entran como features.",
            "Comparacion justa contra 3D Top50: mismos modelos, split temporal y sin tuning.",
            "Se eliminan proxies geograficos red_poblacion_zona_* en vez de conservar uno.",
            "El objetivo es reducir multicolinealidad manteniendo rendimiento operativo.",
        ],
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
