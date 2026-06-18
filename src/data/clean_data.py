from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DIR, RAW_DIR, ensure_dir


@dataclass
class QualityLog:
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, dataset: str, metric: str, before: Any, after: Any, action: str) -> None:
        self.records.append(
            {
                "dataset": dataset,
                "metric": metric,
                "value_before": before,
                "value_after": after,
                "action": action,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / name)


def _month_start(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    return parsed.dt.to_period("M").dt.to_timestamp()


def _mode_or_unknown(series: pd.Series, fallback: str = "Desconocido") -> Any:
    clean = series.dropna()
    if clean.empty:
        return fallback
    mode = clean.mode()
    if mode.empty:
        return clean.iloc[0]
    return mode.iloc[0]


def _deduplicate_exact(df: pd.DataFrame, dataset: str, log: QualityLog) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates().copy()
    log.add(dataset, "duplicados_exactos", before - len(out), 0, "drop_duplicates")
    return out


def _add_missing_flag(df: pd.DataFrame, col: str) -> None:
    df[f"{col}_was_missing"] = df[col].isna().astype(int)


def clean_calidad_senal(log: QualityLog) -> pd.DataFrame:
    dataset = "calidad_senal_zona_mensual"
    df = _deduplicate_exact(_read_csv("calidad_senal_zona_mensual.csv"), dataset, log)
    df["fecha"] = _month_start(df["fecha"])
    log.add(dataset, "fecha_nulos", int(df["fecha"].isna().sum()), 0, "normalizar fecha a mes")
    df = df.dropna(subset=["fecha", "zona_id"]).copy()

    numeric_cols = [
        "poblacion_zona",
        "cobertura_4g_pct",
        "cobertura_5g_pct",
        "latencia_ms",
        "velocidad_media_mbps",
        "tasa_cortes_pct",
        "indice_calidad_global",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["cobertura_4g_pct", "cobertura_5g_pct", "latencia_ms"]:
        _add_missing_flag(df, col)
        before = int(df[col].isna().sum())
        df[col] = df.groupby("tipo_zona", dropna=False)[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())
        log.add(dataset, f"{col}_imputados", before, int(df[col].isna().sum()), "mediana por tipo_zona")

    for col in ["cobertura_4g_pct", "cobertura_5g_pct"]:
        before = int(((df[col] < 0) | (df[col] > 100)).sum())
        df[col] = df[col].clip(0, 100)
        log.add(dataset, f"{col}_fuera_rango", before, 0, "clip 0-100")

    df["incidencia_masiva"] = pd.to_numeric(df["incidencia_masiva"], errors="coerce").fillna(0).clip(0, 1).astype(int)

    agg = {
        "region": _mode_or_unknown,
        "tipo_zona": _mode_or_unknown,
        "poblacion_zona": "median",
        "cobertura_4g_pct": "mean",
        "cobertura_5g_pct": "mean",
        "latencia_ms": "mean",
        "velocidad_media_mbps": "mean",
        "tasa_cortes_pct": "mean",
        "indice_calidad_global": "mean",
        "incidencia_masiva": "max",
        "cobertura_4g_pct_was_missing": "max",
        "cobertura_5g_pct_was_missing": "max",
        "latencia_ms_was_missing": "max",
    }
    before_dups = int(df.duplicated(["zona_id", "fecha"]).sum())
    df = df.groupby(["zona_id", "fecha"], as_index=False).agg(agg)
    log.add(dataset, "duplicados_zona_mes", before_dups, int(df.duplicated(["zona_id", "fecha"]).sum()), "agregacion zona-mes")
    return df


def clean_clientes(log: QualityLog, calidad_clean: pd.DataFrame) -> pd.DataFrame:
    dataset = "clientes"
    df = _deduplicate_exact(_read_csv("clientes.csv"), dataset, log)
    valid_zones = set(calidad_clean["zona_id"].dropna().unique())
    zone_lookup = (
        calidad_clean.sort_values("fecha")
        .groupby("zona_id", as_index=False)
        .agg({"region": _mode_or_unknown, "tipo_zona": _mode_or_unknown, "poblacion_zona": "median"})
    )

    invalid_zone = ~df["zona_id"].isin(valid_zones)
    df["zona_id_was_invalid"] = invalid_zone.astype(int)
    replacement_zone = df.loc[df["zona_id"].isin(valid_zones), "zona_id"].mode().iloc[0]
    before_invalid = int(invalid_zone.sum())
    df.loc[invalid_zone, "zona_id"] = replacement_zone
    log.add(dataset, "zona_id_invalidos", before_invalid, int((~df["zona_id"].isin(valid_zones)).sum()), f"reemplazo por {replacement_zone}")

    df = df.drop(columns=["region", "tipo_zona", "poblacion_zona"], errors="ignore").merge(zone_lookup, on="zona_id", how="left")

    before_id_dups = int(df.duplicated("cliente_id").sum())
    df = df.drop_duplicates(subset=["cliente_id"], keep="first").copy()
    log.add(dataset, "cliente_id_duplicados", before_id_dups, int(df.duplicated("cliente_id").sum()), "mantener primera fila")

    for col in ["edad", "ingreso_estimado", "antiguedad_meses"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        _add_missing_flag(df, col)

    invalid_ant = (df["antiguedad_meses"] < 0)
    df["antiguedad_meses_was_invalid"] = invalid_ant.astype(int)
    df.loc[invalid_ant, "antiguedad_meses"] = np.nan
    log.add(dataset, "antiguedad_negativa", int(invalid_ant.sum()), 0, "marcar e imputar")

    for col in ["edad", "ingreso_estimado", "antiguedad_meses"]:
        before = int(df[col].isna().sum())
        df[col] = df.groupby("tipo_plan", dropna=False)[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())
        log.add(dataset, f"{col}_imputados", before, int(df[col].isna().sum()), "mediana por tipo_plan")

    expected = {
        "sexo": {"F", "M"},
        "estado_civil": {"Soltero/a", "Casado/a", "Divorciado/a"},
        "tipo_plan": {"Prepago", "Contrato", "Premium"},
        "tipo_dispositivo": {"Gama baja", "Gama media", "Gama alta"},
    }
    for col, values in expected.items():
        _add_missing_flag(df, col)
        invalid = ~df[col].isin(values)
        before = int(invalid.sum())
        df.loc[invalid, col] = "Desconocido"
        log.add(dataset, f"{col}_fuera_catalogo", before, int((~df[col].isin(values | {'Desconocido'})).sum()), "Desconocido")

    df["descuento_activo"] = pd.to_numeric(df["descuento_activo"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    return df


def clean_churn_target(log: QualityLog) -> pd.DataFrame:
    dataset = "churn_target"
    df = _deduplicate_exact(_read_csv("churn_target.csv"), dataset, log)
    df["fecha"] = _month_start(df["fecha"])
    df["churn"] = pd.to_numeric(df["churn"], errors="coerce")
    before_invalid = int((~df["churn"].isin([0, 1])).sum())
    df = df.dropna(subset=["cliente_id", "fecha", "churn"]).copy()
    df["churn"] = df["churn"].astype(int)
    df = df[df["churn"].isin([0, 1])].copy()
    before_dups = int(df.duplicated(["cliente_id", "fecha"]).sum())
    df = df.groupby(["cliente_id", "fecha"], as_index=False)["churn"].max()
    log.add(dataset, "churn_fuera_catalogo", before_invalid, int((~df["churn"].isin([0, 1])).sum()), "filtrar binario")
    log.add(dataset, "duplicados_cliente_mes", before_dups, int(df.duplicated(["cliente_id", "fecha"]).sum()), "max churn")
    return df


def clean_facturacion(log: QualityLog) -> pd.DataFrame:
    dataset = "facturacion_mensual"
    df = _deduplicate_exact(_read_csv("facturacion_mensual.csv"), dataset, log)
    df["fecha"] = _month_start(df["fecha"])
    df = df.dropna(subset=["cliente_id", "fecha"]).copy()

    numeric_cols = [
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
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["tipo_plan", "importe_total"]:
        _add_missing_flag(df, col)
        before = int(df[col].isna().sum())
        if col == "tipo_plan":
            df[col] = df.groupby("cliente_id")[col].transform(lambda x: x.ffill().bfill())
            df[col] = df[col].fillna("Desconocido")
        else:
            df[col] = df.groupby("cliente_id")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())
        log.add(dataset, f"{col}_imputados", before, int(df[col].isna().sum()), "cliente y fallback global")

    for col in ["cargo_base", "consumo_extra", "descuento_aplicado", "importe_total", "dias_retraso_pago"]:
        before = int((df[col] < 0).sum())
        df[col] = df[col].clip(lower=0)
        log.add(dataset, f"{col}_negativos", before, int((df[col] < 0).sum()), "clip a cero")

    df["impago_flag"] = df["impago_flag"].fillna(0).clip(0, 1).astype(int)
    df["incidencia_masiva_lag"] = df["incidencia_masiva_lag"].fillna(0).clip(0, 1).astype(int)

    before_dups = int(df.duplicated(["cliente_id", "fecha"]).sum())
    agg = {
        "zona_id": _mode_or_unknown,
        "tipo_plan": _mode_or_unknown,
        "num_lineas": "max",
        "cargo_base": "mean",
        "consumo_extra": "mean",
        "descuento_aplicado": "mean",
        "importe_total": "mean",
        "dias_retraso_pago": "max",
        "impago_flag": "max",
        "variacion_consumo_pct": "mean",
        "stress_calidad_lag": "mean",
        "incidencia_masiva_lag": "max",
        "tipo_plan_was_missing": "max",
        "importe_total_was_missing": "max",
    }
    df = df.groupby(["cliente_id", "fecha"], as_index=False).agg(agg)
    log.add(dataset, "duplicados_cliente_mes", before_dups, int(df.duplicated(["cliente_id", "fecha"]).sum()), "agregacion cliente-mes")
    return df


def clean_interacciones(log: QualityLog) -> pd.DataFrame:
    dataset = "interacciones_soporte"
    df = _deduplicate_exact(_read_csv("interacciones_soporte.csv"), dataset, log)
    before_id_dups = int(df.duplicated("interaccion_id").sum())
    df = df.drop_duplicates(subset=["interaccion_id"], keep="first").copy()
    log.add(dataset, "interaccion_id_duplicados", before_id_dups, int(df.duplicated("interaccion_id").sum()), "mantener primera fila")

    df["fecha_evento"] = pd.to_datetime(df["fecha_evento"], errors="coerce", format="mixed", dayfirst=True)
    df["mes"] = _month_start(df["mes"])
    df = df.dropna(subset=["cliente_id", "mes", "fecha_evento"]).copy()

    df["motivo_was_missing"] = df["motivo"].isna().astype(int)
    df["motivo"] = df["motivo"].fillna("Desconocido")

    for col in ["duracion_min", "satisfaccion_post", "stress_calidad_lag", "dias_retraso_mes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_duration = (df["duracion_min"] < 0)
    df["duracion_min_was_invalid"] = invalid_duration.astype(int)
    df.loc[invalid_duration, "duracion_min"] = np.nan
    before_duration = int(df["duracion_min"].isna().sum())
    df["duracion_min"] = df.groupby("motivo", dropna=False)["duracion_min"].transform(lambda x: x.fillna(x.median()))
    df["duracion_min"] = df["duracion_min"].fillna(df["duracion_min"].median())
    log.add(dataset, "duracion_min_invalidos_o_nulos", before_duration, int(df["duracion_min"].isna().sum()), "mediana por motivo")

    df["satisfaccion_post_was_missing"] = df["satisfaccion_post"].isna().astype(int)
    before_sat = int(df["satisfaccion_post"].isna().sum())
    df["satisfaccion_post"] = df.groupby(["motivo", "resuelto"], dropna=False)["satisfaccion_post"].transform(lambda x: x.fillna(x.median()))
    df["satisfaccion_post"] = df["satisfaccion_post"].fillna(df["satisfaccion_post"].median()).clip(1, 5)
    log.add(dataset, "satisfaccion_post_imputados", before_sat, int(df["satisfaccion_post"].isna().sum()), "mediana por motivo/resuelto")

    df["dias_retraso_mes"] = df["dias_retraso_mes"].fillna(0).clip(lower=0)
    for col in ["resuelto", "impago_mes", "incidencia_masiva_lag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)
    return df


def clean_encuestas(log: QualityLog) -> pd.DataFrame:
    dataset = "encuestas_texto"
    df = _deduplicate_exact(_read_csv("encuestas_texto.csv"), dataset, log)
    df["fecha"] = _month_start(df["fecha"])
    df = df.dropna(subset=["fecha", "zona_id"]).copy()
    df["texto_libre_was_missing"] = df["texto_libre"].isna().astype(int)
    before_text = int(df["texto_libre"].isna().sum())
    df["texto_libre"] = df["texto_libre"].fillna("Sin comentario")
    log.add(dataset, "texto_libre_imputados", before_text, int(df["texto_libre"].isna().sum()), "Sin comentario")

    numeric_cols = [
        "puntuacion_general_1a5",
        "nps_0a10",
        "indice_calidad_global",
        "incidencia_masiva",
        "stress_calidad",
        "flag_incongruente",
        "sent_text_latente",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["puntuacion_general_1a5"] = df["puntuacion_general_1a5"].clip(1, 5)
    df["nps_0a10"] = df["nps_0a10"].clip(0, 10)
    for col in ["incidencia_masiva", "flag_incongruente"]:
        df[col] = df[col].fillna(0).clip(0, 1).astype(int)
    return df


def run_cleaning() -> dict[str, Path]:
    ensure_dir(PROCESSED_DIR)
    log = QualityLog()

    calidad = clean_calidad_senal(log)
    clientes = clean_clientes(log, calidad)
    churn = clean_churn_target(log)
    facturacion = clean_facturacion(log)
    soporte = clean_interacciones(log)
    encuestas = clean_encuestas(log)

    outputs = {
        "calidad_senal_zona_mensual_clean": PROCESSED_DIR / "calidad_senal_zona_mensual_clean.csv",
        "clientes_clean": PROCESSED_DIR / "clientes_clean.csv",
        "churn_target_clean": PROCESSED_DIR / "churn_target_clean.csv",
        "facturacion_mensual_clean": PROCESSED_DIR / "facturacion_mensual_clean.csv",
        "interacciones_soporte_clean": PROCESSED_DIR / "interacciones_soporte_clean.csv",
        "encuestas_texto_clean": PROCESSED_DIR / "encuestas_texto_clean.csv",
        "data_quality_report": PROCESSED_DIR / "data_quality_report.csv",
    }

    calidad.to_csv(outputs["calidad_senal_zona_mensual_clean"], index=False)
    clientes.to_csv(outputs["clientes_clean"], index=False)
    churn.to_csv(outputs["churn_target_clean"], index=False)
    facturacion.to_csv(outputs["facturacion_mensual_clean"], index=False)
    soporte.to_csv(outputs["interacciones_soporte_clean"], index=False)
    encuestas.to_csv(outputs["encuestas_texto_clean"], index=False)
    log.to_frame().to_csv(outputs["data_quality_report"], index=False)

    return outputs


if __name__ == "__main__":
    for name, path in run_cleaning().items():
        print(f"{name}: {path}")
