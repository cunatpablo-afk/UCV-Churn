from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import MODEL_REPORTS_DIR, MODELING_DIR, ensure_dir, project_path

INPUT_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3a.csv"
TOP_FEATURES = MODEL_REPORTS_DIR / "top_features_75_it_3a.csv"
OUTPUT_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3d_top50.csv"
REPORT_CSV = MODELING_DIR / "feature_selection_report_it_3d_top50.csv"
REPORT_JSON = MODELING_DIR / "feature_selection_summary_it_3d_top50.json"

ID_AND_TARGET_COLS = ["cliente_id", "fecha", "churn_t_plus_1"]
TOP_N = 50
GEO_PROXY_TERMS = ["poblacion_zona"]


def build_reduced_dataset() -> dict[str, Path]:
    ensure_dir(MODELING_DIR)
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"No existe {INPUT_DATASET}. Ejecuta primero 3A.")
    if not TOP_FEATURES.exists():
        raise FileNotFoundError(f"No existe {TOP_FEATURES}. Ejecuta primero export_feature_importance_it_3a.py.")

    df = pd.read_csv(INPUT_DATASET, parse_dates=["fecha"])
    top = pd.read_csv(TOP_FEATURES).head(TOP_N)
    top_features = top["feature_original"].tolist()

    missing = [col for col in top_features if col not in df.columns]
    if missing:
        raise ValueError(f"Top features no encontradas en dataset 3A: {missing}")

    selected_columns = ID_AND_TARGET_COLS + top_features
    reduced = df[selected_columns].copy()
    reduced.to_csv(OUTPUT_DATASET, index=False)

    selected_set = set(selected_columns)
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "selected": int(col in selected_set),
                "reason": "top50_it_3a" if col in top_features else "id_target" if col in ID_AND_TARGET_COLS else "fuera_top50",
            }
        )
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False)

    geo_proxy_features = [
        col for col in top_features if any(term in col.lower() for term in GEO_PROXY_TERMS)
    ]
    summary = {
        "iteration": "it_3d_top50",
        "input_dataset": project_path(INPUT_DATASET),
        "output_dataset": project_path(OUTPUT_DATASET),
        "source_importance": project_path(TOP_FEATURES),
        "selection_rule": "top 50 features originales agrupadas desde feature importance de Iteracion 3A",
        "important_notes_for_results_review": [
            "Comparacion justa contra 3C Top75: mismos modelos, mismo split temporal y sin tuning.",
            "Objetivo: comprobar si Top50 mantiene rendimiento con mayor simplicidad.",
            "Si cae poco, Top50 puede ser preferible por interpretabilidad.",
            "Revisar proxies geograficos antes de adopcion final.",
        ],
        "rows": int(len(reduced)),
        "input_columns_total": int(len(df.columns)),
        "selected_columns_total": int(len(selected_columns)),
        "selected_feature_columns": int(len(top_features)),
        "removed_feature_columns": int(len([c for c in df.columns if c not in selected_set and c not in ID_AND_TARGET_COLS])),
        "geo_proxy_features_detected": geo_proxy_features,
        "selected_features": top_features,
        "removed_columns": [c for c in df.columns if c not in selected_set],
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
