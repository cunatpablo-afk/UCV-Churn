from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.paths import MODELING_DIR, PROCESSED_DIR, ensure_dir


TARGET = "churn_t_plus_1"


def _read_processed(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name, parse_dates=["fecha"] if "fecha" in name else None)


def _to_month(df: pd.DataFrame, col: str = "fecha") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return df


def _mode_or_unknown(series: pd.Series, fallback: str = "Desconocido") -> str:
    clean = series.dropna()
    if clean.empty:
        return fallback
    mode = clean.mode()
    return str(mode.iloc[0]) if not mode.empty else str(clean.iloc[0])


def _add_group_lags(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Iterable[str],
    lags: Iterable[int] = (1, 2, 3),
    rolling_window: int = 3,
) -> pd.DataFrame:
    out = df.sort_values([group_col, time_col]).copy()
    for col in value_cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag_{lag}m"] = out.groupby(group_col)[col].shift(lag)
        out[f"{col}_roll_{rolling_window}m"] = (
            out.groupby(group_col)[col]
            .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
        )
    return out


def build_calendar(churn: pd.DataFrame) -> pd.DataFrame:
    churn = _to_month(churn.copy())
    churn = churn.sort_values(["cliente_id", "fecha"]).copy()
    churn["churn"] = churn["churn"].astype(int)
    churn[TARGET] = churn.groupby("cliente_id")["churn"].shift(-1)
    churn["churn_acumulado_hasta_t"] = churn.groupby("cliente_id")["churn"].cumsum()

    out = churn[churn[TARGET].notna()].copy()
    out = out[out["churn_acumulado_hasta_t"] == 0].copy()
    out[TARGET] = out[TARGET].astype(int)
    return out[["cliente_id", "fecha", TARGET]]


def prepare_facturacion() -> pd.DataFrame:
    fact = pd.read_csv(PROCESSED_DIR / "facturacion_mensual_clean.csv")
    fact = _to_month(fact)
    fact = fact.rename(columns={"fecha": "fecha"})
    numeric = [
        "num_lineas",
        "cargo_base",
        "consumo_extra",
        "descuento_aplicado",
        "importe_total",
        "dias_retraso_pago",
        "impago_flag",
        "variacion_consumo_pct",
        "stress_calidad_lag",
        "incidencia_masiva_lag",
        "tipo_plan_was_missing",
        "importe_total_was_missing",
    ]
    fact = fact.rename(columns={c: f"fact_{c}" for c in numeric if c in fact.columns})
    fact = fact.rename(columns={"tipo_plan": "fact_tipo_plan", "zona_id": "fact_zona_id"})
    value_cols = [f"fact_{c}" for c in numeric if f"fact_{c}" in fact.columns]
    fact = _add_group_lags(fact, "cliente_id", "fecha", value_cols)
    return fact


def prepare_calidad() -> pd.DataFrame:
    calidad = pd.read_csv(PROCESSED_DIR / "calidad_senal_zona_mensual_clean.csv")
    calidad = _to_month(calidad)
    numeric = [
        "poblacion_zona",
        "cobertura_4g_pct",
        "cobertura_5g_pct",
        "latencia_ms",
        "velocidad_media_mbps",
        "tasa_cortes_pct",
        "indice_calidad_global",
        "incidencia_masiva",
        "cobertura_4g_pct_was_missing",
        "cobertura_5g_pct_was_missing",
        "latencia_ms_was_missing",
    ]
    calidad = calidad.rename(columns={c: f"red_{c}" for c in numeric if c in calidad.columns})
    calidad = calidad.rename(columns={"region": "red_region", "tipo_zona": "red_tipo_zona"})
    value_cols = [f"red_{c}" for c in numeric if f"red_{c}" in calidad.columns]
    calidad = _add_group_lags(calidad, "zona_id", "fecha", value_cols)
    return calidad


def prepare_soporte() -> pd.DataFrame:
    soporte = pd.read_csv(PROCESSED_DIR / "interacciones_soporte_clean.csv")
    soporte["mes"] = pd.to_datetime(soporte["mes"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    agg = (
        soporte.groupby(["cliente_id", "mes"], as_index=False)
        .agg(
            soporte_contactos=("interaccion_id", "count"),
            soporte_no_resueltos=("resuelto", lambda x: int((x == 0).sum())),
            soporte_resuelto_rate=("resuelto", "mean"),
            soporte_satisfaccion_media=("satisfaccion_post", "mean"),
            soporte_satisfaccion_min=("satisfaccion_post", "min"),
            soporte_duracion_media=("duracion_min", "mean"),
            soporte_duracion_max=("duracion_min", "max"),
            soporte_impago_mes=("impago_mes", "max"),
            soporte_dias_retraso_media=("dias_retraso_mes", "mean"),
            soporte_stress_calidad_media=("stress_calidad_lag", "mean"),
            soporte_incidencia_masiva=("incidencia_masiva_lag", "max"),
            soporte_motivo_principal=("motivo", _mode_or_unknown),
            soporte_canal_principal=("canal", _mode_or_unknown),
            soporte_motivo_missing_rate=("motivo_was_missing", "mean"),
            soporte_satisfaccion_missing_rate=("satisfaccion_post_was_missing", "mean"),
        )
        .rename(columns={"mes": "fecha"})
    )
    value_cols = [
        "soporte_contactos",
        "soporte_no_resueltos",
        "soporte_resuelto_rate",
        "soporte_satisfaccion_media",
        "soporte_satisfaccion_min",
        "soporte_duracion_media",
        "soporte_duracion_max",
        "soporte_impago_mes",
        "soporte_dias_retraso_media",
        "soporte_stress_calidad_media",
        "soporte_incidencia_masiva",
        "soporte_motivo_missing_rate",
        "soporte_satisfaccion_missing_rate",
    ]
    return _add_group_lags(agg, "cliente_id", "fecha", value_cols)


def prepare_encuestas() -> pd.DataFrame:
    encuestas = pd.read_csv(PROCESSED_DIR / "encuestas_texto_clean.csv")
    encuestas = _to_month(encuestas)
    agg = (
        encuestas.groupby(["zona_id", "fecha"], as_index=False)
        .agg(
            encuestas_n=("encuesta_id", "count"),
            encuestas_puntuacion_media=("puntuacion_general_1a5", "mean"),
            encuestas_nps_media=("nps_0a10", "mean"),
            encuestas_sentimiento_medio=("sent_text_latente", "mean"),
            encuestas_stress_medio=("stress_calidad", "mean"),
            encuestas_incongruentes_rate=("flag_incongruente", "mean"),
            encuestas_texto_missing_rate=("texto_libre_was_missing", "mean"),
        )
    )
    value_cols = [c for c in agg.columns if c not in ["zona_id", "fecha"]]
    return _add_group_lags(agg, "zona_id", "fecha", value_cols)


def run_feature_engineering() -> dict[str, Path]:
    ensure_dir(MODELING_DIR)

    churn = pd.read_csv(PROCESSED_DIR / "churn_target_clean.csv")
    calendar = build_calendar(churn)
    clientes = pd.read_csv(PROCESSED_DIR / "clientes_clean.csv")
    fact = prepare_facturacion()
    calidad = prepare_calidad()
    soporte = prepare_soporte()
    encuestas = prepare_encuestas()

    dataset = calendar.merge(clientes, on="cliente_id", how="left")
    dataset = dataset.merge(fact, on=["cliente_id", "fecha"], how="left")
    dataset = dataset.merge(calidad, on=["zona_id", "fecha"], how="left")
    dataset = dataset.merge(soporte, on=["cliente_id", "fecha"], how="left")
    dataset = dataset.merge(encuestas, on=["zona_id", "fecha"], how="left")

    zero_fill = [
        c
        for c in dataset.columns
        if c.startswith("soporte_contactos")
        or c.startswith("soporte_no_resueltos")
        or c.startswith("soporte_impago_mes")
        or c.startswith("soporte_incidencia_masiva")
        or c.startswith("encuestas_n")
    ]
    for col in zero_fill:
        dataset[col] = dataset[col].fillna(0)

    forbidden = {"churn", "ever_churn"}
    forbidden_present = sorted(forbidden.intersection(dataset.columns))
    if forbidden_present:
        raise ValueError(f"Columnas prohibidas presentes en dataset: {forbidden_present}")

    dataset = dataset.sort_values(["fecha", "cliente_id"]).copy()
    output = MODELING_DIR / "churn_modeling_dataset.csv"
    quality_output = MODELING_DIR / "modeling_quality_report.csv"

    quality = pd.DataFrame(
        [
            {"metric": "filas", "value": len(dataset)},
            {"metric": "columnas", "value": dataset.shape[1]},
            {"metric": "clientes_unicos", "value": dataset["cliente_id"].nunique()},
            {"metric": "mes_min", "value": dataset["fecha"].min()},
            {"metric": "mes_max", "value": dataset["fecha"].max()},
            {"metric": "target_nulos", "value": int(dataset[TARGET].isna().sum())},
            {"metric": "target_fuera_catalogo", "value": int((~dataset[TARGET].isin([0, 1])).sum())},
            {"metric": "duplicados_cliente_mes", "value": int(dataset.duplicated(["cliente_id", "fecha"]).sum())},
            {"metric": "columnas_prohibidas_presentes", "value": ",".join(forbidden_present)},
            {"metric": "tasa_churn_t_plus_1", "value": float(dataset[TARGET].mean())},
        ]
    )

    dataset.to_csv(output, index=False)
    quality.to_csv(quality_output, index=False)
    return {"dataset": output, "quality_report": quality_output}


if __name__ == "__main__":
    for name, path in run_feature_engineering().items():
        print(f"{name}: {path}")
