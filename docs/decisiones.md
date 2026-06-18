# Registro De Decisiones

## Decision 001 - Unidad De Prediccion

- Fecha: 2026-06-16
- Decision: usar `cliente_id` + `fecha` mensual como unidad base.
- Motivo: el objetivo de negocio es anticipar churn a un mes vista y las fuentes principales tienen granularidad mensual o pueden agregarse a mes.
- Riesgo mitigado: evita convertir el problema en `ever_churn`, que introduce perdida temporal y riesgo de leakage.

## Decision 002 - Target

- Fecha: 2026-06-16
- Decision: usar `churn_t_plus_1`, creado con el churn del mes siguiente por cliente.
- Motivo: simula una prediccion accionable antes de que ocurra la baja.
- Restriccion: ninguna variable predictora puede usar informacion posterior al mes `t`.

## Decision 003 - EDA

- Fecha: 2026-06-16
- Decision: rehacer los EDA sobre `data/processed` en notebooks tematicos.
- Motivo: los notebooks antiguos sirven como exploracion, pero mezclan rutas antiguas, datos raw y modelado dentro del EDA.

## Decision 004 - Validacion De Modelos

- Fecha: 2026-06-16
- Decision: usar test temporal final y validacion estratificada solo dentro de entrenamiento.
- Motivo: el problema es temporal, pero tambien hay desbalance de clase.

## Decision 005 - Seleccion De Modelo

- Fecha: 2026-06-16
- Decision: priorizar PR-AUC, Recall y F1 de churn sobre Accuracy.
- Motivo: churn es una clase minoritaria y Accuracy puede favorecer modelos inutiles para retencion.

## Decision 006 - EDA Rehecho

- Fecha: 2026-06-16
- Decision: crear notebooks EDA nuevos, numerados y tematicos, usando `data/processed` y `data/modeling`.
- Notebooks:
  - `01_EDA_overview_calidad_datos.ipynb`
  - `02_EDA_clientes_churn.ipynb`
  - `03_EDA_facturacion_churn.ipynb`
  - `04_EDA_calidad_senal_churn.ipynb`
  - `05_EDA_soporte_churn.ipynb`
  - `06_EDA_encuestas_percepcion.ipynb`
  - `07_EDA_integrado_pre_modelado.ipynb`
- Motivo: los notebooks antiguos sirven como fuente de ideas, pero mezclaban EDA, datos raw y modelado.
- Validacion: los 7 notebooks se ejecutaron con un runner ligero sin errores.

## Decision 007 - Lectura Del Benchmark De Modelos

- Fecha: 2026-06-16
- Decision: mantener `gradient_boosting` como ganador tecnico por ranking automatico, pero no considerarlo aun una recomendacion operativa cerrada.
- Motivo: obtiene el mayor PR-AUC en test temporal, pero con threshold 0.5 detecta muy pocos churners.
- Implicacion: antes de recomendar una accion de retencion hay que optimizar threshold y definir un coste de falsos negativos/falsos positivos.
- Alternativa operativa inicial: revisar `logistic_regression`, que tiene menor PR-AUC pero mayor recall de churn.
