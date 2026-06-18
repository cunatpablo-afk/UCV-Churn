# Informe De Modelado Churn

## 1. Objetivo Del Proyecto

El objetivo del proyecto es construir un modelo de prediccion de churn para clientes, con una logica temporal realista y accionable por negocio.

La unidad de analisis definida es:

```text
cliente_id + fecha mensual
```

El target final es:

```text
churn_t_plus_1
```

Esto significa que, para cada cliente y mes `t`, el modelo intenta predecir si el cliente hara churn en el mes siguiente `t+1`, usando solo informacion disponible hasta el mes `t`.

Esta decision evita convertir el problema en un `ever_churn` historico y reduce el riesgo de leakage temporal.

## 2. Pipeline Construido

El proyecto se estructuro como un flujo end-to-end:

1. Limpieza reproducible de datos raw.
2. Generacion de datos procesados.
3. Construccion del dataset cliente-mes.
4. Rehacer EDA sobre datos procesados.
5. Entrenamiento de modelos con split temporal.
6. Seleccion iterativa de variables.
7. Evaluacion operativa mediante lift y politicas Top-K.
8. Analisis de overfitting, multicolinealidad y explicabilidad.

Archivos principales:

```text
src/data/clean_data.py
src/features/build_modeling_dataset.py
src/models/train_models.py
docs/modelos.md
docs/decisiones.md
```

## 3. Limpieza Y Datos Procesados

La limpieza se implemento en:

```text
src/data/clean_data.py
```

Salidas principales:

```text
data/processed/clientes_clean.csv
data/processed/churn_target_clean.csv
data/processed/facturacion_mensual_clean.csv
data/processed/calidad_senal_zona_mensual_clean.csv
data/processed/interacciones_soporte_clean.csv
data/processed/encuestas_texto_clean.csv
data/processed/data_quality_report.csv
```

Acciones principales:

- Normalizacion de fechas a mes.
- Eliminacion de duplicados exactos.
- Agregacion a granularidad cliente-mes o zona-mes cuando correspondia.
- Correccion de valores imposibles.
- Imputaciones justificadas con flags cuando la ausencia podia ser informativa.
- No se modificaron los datos raw.

## 4. Dataset De Modelado

El dataset base de modelado se genero en:

```text
data/modeling/churn_modeling_dataset.csv
```

Resumen de calidad:

| Metrica | Valor |
|---|---:|
| Filas | 311987 |
| Columnas | 246 |
| Clientes unicos | 9982 |
| Mes minimo | 2023-01-01 |
| Mes maximo | 2025-11-01 |
| Target nulos | 0 |
| Target fuera de catalogo | 0 |
| Duplicados cliente-mes | 0 |
| Tasa churn_t_plus_1 | 0.6353% |

El target esta muy desbalanceado. En el test temporal final la tasa de churn es aproximadamente:

```text
0.5155%
```

Por este motivo, Accuracy no se usa como metrica principal.

## 5. Estrategia De Validacion

La evaluacion principal usa split temporal.

Ventana de test:

```text
2025-06-01 a 2025-11-01
```

Resumen:

| Split | Filas |
|---|---:|
| Train | 263099 |
| Test temporal | 48888 |

Distribucion en test temporal:

| Clase | Filas |
|---|---:|
| No churn | 48636 |
| Churn | 252 |

Se usa validacion cruzada estratificada dentro de train como apoyo, pero la decision principal se toma con el test temporal.

## 6. Modelos Probados

Se probaron varias familias de modelos:

- Dummy baseline.
- Logistic Regression.
- Random Forest.
- Gradient Boosting.
- XGBoost.

Con el avance del proyecto se priorizaron:

- Logistic Regression, por estabilidad, interpretabilidad y mejor PR-AUC.
- XGBoost, como alternativa no lineal.

La metrica principal de seleccion fue:

```text
PR-AUC en test temporal
```

Metricas secundarias:

- ROC-AUC.
- Precision.
- Recall.
- F1.
- Matriz de confusion.

## 7. Evolucion De Iteraciones

Resumen de las iteraciones principales:

| Iteracion | Dataset | Features | Mejor modelo | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 2 | Completo | 243 | Logistic Regression | 0.0237 | 0.6983 | 0.5119 | 0.0129 | 0.0253 |
| 3A | Sin geografia directa | 183 | Logistic Regression | 0.0238 | 0.6992 | 0.5238 | 0.0126 | 0.0246 |
| 3B | Geo agregada | 231 | Logistic Regression | 0.0227 | 0.6963 | 0.5278 | 0.0124 | 0.0243 |
| 3C | Top75 | 75 | Logistic Regression | 0.0264 | 0.7085 | 0.5079 | 0.0129 | 0.0251 |
| 3D | Top50 | 50 | Logistic Regression | 0.0266 | 0.7130 | 0.5317 | 0.0131 | 0.0256 |
| 3E | Top30 | 30 | Logistic Regression | 0.0241 | 0.7120 | 0.4960 | 0.0142 | 0.0276 |
| 3F | Poda por familias | 37 | Logistic Regression | 0.0267 | 0.7132 | 0.5238 | 0.0136 | 0.0266 |
| 3G | Imputacion semantica | 37 | Logistic Regression | 0.0267 | 0.7132 | 0.5238 | 0.0136 | 0.0265 |

Lectura:

- 3B no mejoro 3A, por lo que no se adopto la geografia agregada.
- 3C mejoro al reducir a Top75.
- 3D Top50 fue una mejora clara frente a 3C.
- 3E Top30 simplifico demasiado y perdio PR-AUC/recall.
- 3F elimino redundancias y proxies geograficos, manteniendo rendimiento.
- 3G cambio la imputacion para Logistic y obtuvo una ligera mejora adicional.

## 8. Mejor Iteracion Actual

La mejor iteracion actual es:

```text
Iteracion 3G - Logistic Regression con imputacion semantica
```

Esta iteracion usa el dataset de 3F:

```text
data/modeling/churn_modeling_dataset_it_3f_family_pruning.csv
```

El dataset 3F contiene:

| Elemento | Valor |
|---|---:|
| Filas | 311987 |
| Columnas totales | 40 |
| Features de modelo | 37 |

3F parte del Top50 y elimina redundancias por familias:

- Importes redundantes.
- Cargo base redundante.
- Numero de lineas redundante.
- Indices de calidad redundantes.
- Proxies geograficos `red_poblacion_zona_*`.

3G mantiene ese dataset, pero cambia el preprocesado de Logistic:

- Numericas de eventos, conteos y flags: imputacion a 0.
- Resto de numericas: mediana.
- Categoricas: `"Desconocido"`.
- Numericas con `add_indicator=True`.
- Escalado con `StandardScaler`.
- Categoricas con one-hot encoding.

## 9. Metricas De La Mejor Iteracion

Modelo:

```text
Logistic Regression
```

Configuracion:

```text
C = 0.1
solver = lbfgs
class_weight = balanced
max_iter = 5000
threshold = 0.5
```

Metricas en test temporal:

| Metrica | Valor |
|---|---:|
| PR-AUC | 0.0267 |
| ROC-AUC | 0.7132 |
| Accuracy | 0.8019 |
| Precision | 0.0136 |
| Recall | 0.5238 |
| F1 | 0.0265 |

Comparacion contra dummy:

| Modelo | PR-AUC test | ROC-AUC test | Recall |
|---|---:|---:|---:|
| Dummy | 0.0052 | 0.5000 | 0.0000 |
| Logistic 3G | 0.0267 | 0.7132 | 0.5238 |

El modelo multiplica aproximadamente por 5.17 el PR-AUC del baseline dummy:

```text
0.0267 / 0.0052 ~= 5.17
```

## 10. Matriz De Confusion

Matriz de confusion en test temporal:

```text
[[39073, 9563],
 [  120,  132]]
```

Interpretacion:

| Concepto | Valor |
|---|---:|
| TN - No churn bien clasificados | 39073 |
| FP - No churn marcados como riesgo | 9563 |
| FN - Churn no detectados | 120 |
| TP - Churn detectados | 132 |

Resumen operativo:

| Metrica operativa | Valor |
|---|---:|
| Churners reales en test | 252 |
| Churners detectados | 132 |
| Churners no detectados | 120 |
| Clientes contactados por threshold 0.5 | 9695 |
| Falsos positivos | 9563 |

Lectura:

- El modelo detecta algo mas de la mitad de los churners reales.
- El coste es una cantidad elevada de falsos positivos.
- Por eso el modelo debe entenderse principalmente como un sistema de ranking/priorizacion, no como clasificador binario cerrado con threshold 0.5.

## 11. Analisis De Overfitting

Diagnostico de overfitting para la mejor iteracion:

| Split | ROC-AUC | PR-AUC | Recall | F1 |
|---|---:|---:|---:|---:|
| Train | 0.7043 | 0.0218 | 0.5786 | 0.0260 |
| CV media | 0.6910 | 0.0211 | 0.5711 | 0.0255 |
| Test temporal | 0.7132 | 0.0267 | 0.5238 | 0.0265 |

Gaps:

| Gap | Valor |
|---|---:|
| Train - CV ROC-AUC | 0.0132 |
| Train - Test ROC-AUC | -0.0089 |
| CV - Test ROC-AUC | -0.0222 |
| Train - CV PR-AUC | 0.0007 |
| Train - Test PR-AUC | -0.0049 |
| CV - Test PR-AUC | -0.0056 |

Diagnostico:

```text
ok
```

No se observa overfitting grave.

La metrica de test temporal incluso queda por encima de train/CV en PR-AUC. Esto puede ocurrir porque la ventana temporal final tiene una distribucion algo mas favorable para el ranking, pero no indica sobreajuste.

## 12. Evaluacion Operativa

Ademas del threshold fijo, se evaluo el modelo como ranking de riesgo mediante Top-K.

Resultados principales:

| Politica | Contactados | Churners capturados | Precision@K | Capture rate | Lift |
|---|---:|---:|---:|---:|---:|
| Top 1% | 489 | 22 | 0.0450 | 0.0873 | 8.73 |
| Top 2% | 978 | 39 | 0.0399 | 0.1548 | 7.74 |
| Top 5% | 2444 | 68 | 0.0278 | 0.2698 | 5.40 |
| Top 10% | 4889 | 95 | 0.0194 | 0.3770 | 3.77 |
| Top 20% | 9778 | 128 | 0.0131 | 0.5079 | 2.54 |
| Top 30% | 14666 | 154 | 0.0105 | 0.6111 | 2.04 |

Lectura:

- El Top 1% concentra clientes con una tasa de churn 8.73 veces superior a una seleccion aleatoria.
- El Top 5% captura el 26.98% de los churners contactando solo 2444 clientes.
- El Top 10% captura el 37.70% de los churners y es una politica equilibrada.
- El Top 20% captura aproximadamente la mitad de los churners, pero con alto volumen de falsos positivos.

Recomendacion operativa:

- Top 5% para piloto conservador.
- Top 10% como politica equilibrada.
- Top 20% solo si negocio acepta contactar mucho volumen.

## 13. Fortalezas Del Proyecto

Fortalezas principales:

- Target temporal bien definido.
- Test temporal, no split aleatorio.
- Separacion entre limpieza, feature engineering y modelado.
- Control explicito de columnas prohibidas.
- Registro vivo de decisiones y experimentos.
- Evaluacion con PR-AUC y no con Accuracy.
- Analisis operativo Top-K y lift.
- Reduccion progresiva de variables.
- Eliminacion de proxies geograficos claros en 3F.
- Comparacion contra dummy baseline.

## 14. Limitaciones Actuales

Limitaciones a tener en cuenta:

1. El test temporal se ha usado muchas veces para tomar decisiones.

   Esto puede introducir cierto sobreajuste metodologico al test. Antes de cerrar el proyecto, conviene reservar un ultimo holdout temporal intocable.

2. El modelo usa informacion del mes `t`.

   Esto es correcto si el modelo se ejecuta al cierre del mes. Debe explicitarse en la documentacion.

3. La precision es baja.

   Esto es esperable por la tasa mensual de churn, pero implica que el uso operativo debe ser Top-K y no clasificacion binaria ingenua.

4. Sigue existiendo multicolinealidad residual.

   3F reduce redundancias importantes, pero variables de red/calidad y retrasos siguen correlacionadas.

5. Logistic es interpretable, pero sus coeficientes deben leerse con cautela.

   Debido a one-hot, escalado, indicadores de nulos y correlaciones, no conviene hacer interpretaciones causales directas.

## 15. Decision Actual

La version recomendada actualmente es:

```text
Modelo: Logistic Regression
Iteracion: 3G
Dataset: churn_modeling_dataset_it_3f_family_pruning.csv
Uso: ranking de riesgo y seleccion Top-K
```

Motivos:

- Mejor PR-AUC observado.
- ROC-AUC estable.
- Recall razonable para un problema muy desbalanceado.
- Dataset simplificado a 37 features.
- Sin proxies geograficos `red_poblacion_zona_*`.
- Sin evidencia de overfitting grave.
- Mas defendible que modelos con metricas aparentemente mejores pero con riesgo de leakage o split aleatorio.

## 16. Proximos Pasos Recomendados

Antes de la entrega final:

1. Ordenar `docs/modelos.md`.
2. Crear un holdout temporal final realmente intocable.
3. Recalcular lift/Top-K sobre la mejor version 3G, no solo sobre 3C.
4. Documentar claramente el momento de scoring: cierre del mes `t`.
5. Preparar una tabla final con:
   - modelo elegido,
   - variables usadas,
   - metricas,
   - matriz de confusion,
   - politica operativa recomendada.

Conclusion:

El proyecto no vende metricas artificialmente altas. Presenta un modelo modesto pero metodologicamente defendible, con utilidad real como herramienta de priorizacion de clientes en riesgo.

## 17. Factores Mas Importantes Asociados Al Churn

La importancia de variables debe interpretarse con cautela. El modelo principal es una regresion logistica con variables escaladas, one-hot encoding, indicadores de nulos y algunas familias de variables correlacionadas. Por tanto, estas variables no deben leerse como causas directas del churn, sino como senales predictivas utiles dentro del modelo.

Las variables mas relevantes de la Iteracion 3G se agrupan en cuatro grandes familias:

### 17.1. Problemas De Calidad Y Red

Variables destacadas:

```text
fact_stress_calidad_lag_roll_3m
fact_stress_calidad_lag_lag_1m
fact_stress_calidad_lag_lag_2m
fact_stress_calidad_lag
red_indice_calidad_global_roll_3m
red_cobertura_5g_pct_lag_2m
red_cobertura_5g_pct_lag_3m
red_cobertura_4g_pct_lag_1m
red_cobertura_4g_pct_lag_3m
red_tasa_cortes_pct_lag_3m
```

Lectura de negocio:

- Los clientes expuestos a peor calidad de red o deterioro sostenido presentan mayor riesgo de churn.
- La persistencia de la mala calidad durante varios meses parece mas informativa que una unica observacion aislada.
- La cobertura 4G/5G y la tasa de cortes aportan informacion contextual sobre la experiencia tecnica del cliente.

Implicacion:

Las acciones de retencion no deberian limitarse a descuentos. Para clientes con senales tecnicas, una accion mas adecuada puede ser revision de cobertura, soporte tecnico proactivo o comunicacion especifica sobre incidencias.

### 17.2. Retrasos De Pago E Impagos

Variables destacadas:

```text
fact_dias_retraso_pago_roll_3m
fact_dias_retraso_pago_lag_1m
fact_dias_retraso_pago_lag_2m
fact_dias_retraso_pago
fact_impago_flag_roll_3m
fact_impago_flag_lag_2m
```

Lectura de negocio:

- Los retrasos de pago son una de las familias mas fuertes del modelo.
- El comportamiento reciente y acumulado de retrasos aporta senal relevante.
- Los impagos no solo reflejan riesgo financiero; tambien pueden estar relacionados con insatisfaccion, abandono progresivo o baja vinculacion.

Implicacion:

Clientes con retrasos o impagos recurrentes podrian requerir acciones diferenciadas: regularizacion flexible, revision de factura, comunicacion preventiva o propuesta de plan mas ajustado.

### 17.3. Soporte Y Relacion Con El Cliente

Variables destacadas:

```text
soporte_canal_principal
soporte_motivo_principal
soporte_duracion_media_roll_3m
soporte_duracion_max_roll_3m
soporte_impago_mes_roll_3m
soporte_satisfaccion_min
soporte_satisfaccion_media
soporte_satisfaccion_media_roll_3m
soporte_satisfaccion_min_roll_3m
soporte_dias_retraso_media_roll_3m
```

Lectura de negocio:

- El motivo y el canal de soporte contienen informacion relevante sobre el riesgo.
- La satisfaccion posterior al soporte ayuda a distinguir clientes con experiencia deteriorada.
- Duraciones altas o incidencias repetidas pueden capturar friccion operativa.

Implicacion:

El modelo sugiere que la gestion relacional importa. Un cliente con contactos recientes, baja satisfaccion o motivos sensibles debe ser priorizado de forma distinta a un cliente sin actividad de soporte.

### 17.4. Facturacion, Plan Y Vinculacion

Variables destacadas:

```text
fact_tipo_plan
fact_cargo_base_lag_1m
fact_descuento_aplicado_roll_3m
fact_descuento_aplicado_lag_2m
fact_importe_total_roll_3m
fact_num_lineas
```

Lectura de negocio:

- El tipo de plan y la estructura de facturacion ayudan a segmentar riesgo.
- El importe medio reciente aporta informacion, pero tras la poda por familias se evita mantener muchas versiones redundantes del mismo concepto.
- El numero de lineas puede actuar como proxy de vinculacion: clientes con mas lineas pueden comportarse distinto a clientes con una unica linea.

Implicacion:

La estrategia de retencion deberia considerar el valor y perfil del cliente, no solo su score. Un mismo score puede requerir acciones distintas segun plan, importe y vinculacion.

### 17.5. Resumen De Factores

Los factores mas importantes del churn en este proyecto son:

1. Deterioro o mala calidad de red.
2. Retrasos de pago e impagos.
3. Contactos con soporte, motivo de contacto y satisfaccion.
4. Tipo de plan, facturacion reciente y vinculacion del cliente.

La lectura global es coherente desde negocio: el churn parece asociado a una combinacion de problemas tecnicos, friccion economica y experiencia de soporte.

## 18. Como Funciona El Dataset Temporal

El dataset de modelado no es un dataset clasico de una fila por cliente. Es un panel temporal mensual.

Cada fila representa:

```text
un cliente en un mes concreto
```

La clave de cada fila es:

```text
cliente_id + fecha
```

Ejemplo:

| cliente_id | fecha | Significado |
|---|---|---|
| C001 | 2025-03-01 | Estado del cliente C001 durante marzo de 2025 |
| C001 | 2025-04-01 | Estado del cliente C001 durante abril de 2025 |
| C001 | 2025-05-01 | Estado del cliente C001 durante mayo de 2025 |

El target se construye desplazando el churn un mes hacia delante:

```text
churn_t_plus_1 = churn del cliente en el mes siguiente
```

Ejemplo simplificado:

| cliente_id | fecha | churn del mes t | churn_t_plus_1 |
|---|---|---:|---:|
| C001 | 2025-03-01 | 0 | 0 |
| C001 | 2025-04-01 | 0 | 1 |
| C001 | 2025-05-01 | 1 | sin uso |

La fila importante para prediccion es abril de 2025:

```text
Con informacion disponible hasta abril, el modelo aprende que en mayo el cliente hizo churn.
```

Las filas posteriores al primer churn del cliente se eliminan, porque despues de churn el cliente ya no es un candidato realista para retencion.

### 18.1. Que Informacion Puede Usar El Modelo

Para una fila `(cliente_id, fecha=t)`, el modelo puede usar:

- informacion estatica del cliente;
- facturacion del mes `t`;
- soporte del mes `t`;
- calidad de red del mes `t`;
- lags de meses anteriores, como `t-1`, `t-2`, `t-3`;
- rollings calculados hasta `t`.

No puede usar:

- churn del mes `t+1` como feature;
- informacion posterior a `t`;
- identificadores directos como `cliente_id`;
- variables tipo `ever_churn`;
- fechas futuras.

Por eso `cliente_id`, `fecha` y `churn_t_plus_1` se conservan en el CSV para trazabilidad, pero se eliminan de `X_train` y `X_test` antes de entrenar.

### 18.2. Ejemplo De Cliente Nuevo

Supongamos un cliente nuevo que aparece por primera vez en marzo:

| cliente_id | fecha | importe_total | soporte_contactos | importe_lag_1m | importe_lag_2m | importe_lag_3m |
|---|---|---:|---:|---:|---:|---:|
| C999 | 2025-03-01 | 35.0 | 0 | NaN | NaN | NaN |
| C999 | 2025-04-01 | 37.0 | 1 | 35.0 | NaN | NaN |
| C999 | 2025-05-01 | 36.0 | 0 | 37.0 | 35.0 | NaN |
| C999 | 2025-06-01 | 39.0 | 2 | 36.0 | 37.0 | 35.0 |

En marzo, el cliente no tiene historial previo. Por eso sus lags aparecen como nulos.

Esto no es necesariamente un error: la ausencia de historial tambien es informacion. Para conservarla, el pipeline de modelado usa:

```text
add_indicator=True
```

Esto crea indicadores que marcan si una variable era nula antes de imputar. Asi el modelo puede aprender diferencias entre:

- cliente con valor real bajo;
- cliente nuevo sin historial suficiente.

En la Iteracion 3G se refino la imputacion:

- conteos, eventos y flags se imputan a 0 cuando tiene sentido de negocio;
- variables continuas como importes, calidad o retrasos se imputan con mediana;
- categoricas se imputan como `"Desconocido"`.

### 18.3. Momento De Uso Del Modelo

El modelo debe ejecutarse al cierre del mes `t`.

Ejemplo:

```text
Fecha de scoring: cierre de abril 2025
Informacion disponible: datos hasta abril 2025
Prediccion: probabilidad de churn en mayo 2025
```

Esto es importante porque el modelo usa variables del mes `t`. Si se quisiera predecir al inicio del mes, habria que cambiar el dataset para usar solo informacion hasta `t-1`.

### 18.4. Por Que Este Enfoque Es Mas Realista

Este diseno temporal es mas exigente que un split aleatorio porque simula mejor el uso real:

```text
entrenar con pasado -> predecir futuro
```

Ademas, evita mezclar meses futuros dentro del entrenamiento y reduce el riesgo de obtener metricas artificialmente optimistas.
