from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import MODELING_DIR, ensure_dir, project_path

INPUT_DATASET = MODELING_DIR / "churn_modeling_dataset.csv"
OUTPUT_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3a.csv"
REPORT_CSV = MODELING_DIR / "feature_selection_report_it_3a.csv"
REPORT_JSON = MODELING_DIR / "feature_selection_summary_it_3a.json"

ID_AND_TARGET_COLS = ["cliente_id", "fecha", "churn_t_plus_1"]
DOMAIN_PREFIXES = ("fact_", "soporte_", "red_")

## Iteracion 3A:
## Conservamos solo variables de facturacion, soporte y calidad de red.
## Excluimos variables geograficas directas para medir cuanto rendimiento venia
## de comportamiento/servicio frente a ubicacion.
GEO_EXACT_EXCLUDE = {
    "zona_id",
    "fact_zona_id",
    "region",
    "red_region",
    "tipo_zona",
    "red_tipo_zona",
}
GEO_KEYWORDS_EXCLUDE = (
    "zona_id",
    "region",
    "tipo_zona",
    "poblacion_zona",
)


def classify_column(column: str) -> tuple[bool, str, str]:
    if column in ID_AND_TARGET_COLS:
        return True, "id_target", "columna de trazabilidad/target"

    if column in GEO_EXACT_EXCLUDE:
        return False, "geografia", "exclusion geografica explicita"

    if any(keyword in column for keyword in GEO_KEYWORDS_EXCLUDE):
        return False, "geografia", "exclusion por patron geografico"

    if column.startswith(DOMAIN_PREFIXES):
        domain = column.split("_", 1)[0]
        return True, domain, "dominio permitido por prefijo"

    return False, "fuera_dominio", "no pertenece a facturacion/soporte/red"


def build_reduced_dataset() -> dict[str, Path]:
    ensure_dir(MODELING_DIR)
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"No existe {INPUT_DATASET}. Ejecuta primero feature engineering.")

    df = pd.read_csv(INPUT_DATASET, parse_dates=["fecha"])
    report_rows = []
    selected_columns = []

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

    reduced = df[selected_columns].copy()
    reduced.to_csv(OUTPUT_DATASET, index=False)

    report = pd.DataFrame(report_rows)
    report.to_csv(REPORT_CSV, index=False)

    selected_features = [c for c in selected_columns if c not in ID_AND_TARGET_COLS]
    removed_features = [c for c in df.columns if c not in selected_columns and c not in ID_AND_TARGET_COLS]
    summary = {
        "iteration": "it_3a",
        "input_dataset": project_path(INPUT_DATASET),
        "output_dataset": project_path(OUTPUT_DATASET),
        "selection_rule": "prefijos fact_, soporte_, red_ con blacklist geografica explicita",
        "important_notes_for_results_review": [
            "Iteracion 3A excluye variables geograficas directas.",
            "Comparar despues con Iteracion 3B: geografia agregada no identificativa, sin IDs directos.",
            "La seleccion por nombre es practica pero imperfecta; revisar manualmente columnas conservadas/eliminadas.",
            "Comparacion principal: mismos hiperparametros de Iteracion 2.",
            "Comparacion secundaria opcional: pequeno tuning de XGBoost reducido si el rendimiento cae o queda cerca.",
        ],
        "rows": int(len(reduced)),
        "original_columns_total": int(len(df.columns)),
        "selected_columns_total": int(len(selected_columns)),
        "selected_feature_columns": int(len(selected_features)),
        "removed_feature_columns": int(len(removed_features)),
        "selected_columns": selected_columns,
        "removed_columns": [c for c in df.columns if c not in selected_columns],
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
