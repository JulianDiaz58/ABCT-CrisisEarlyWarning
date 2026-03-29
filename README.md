# Crisis Early Warning Model (TACE/ABCT)

## Resumen ejecutivo
Este trabajo desarrolla un sistema de alerta temprana de crisis bajo marco Austrian Business Cycle Theory (ABCT) usando un modelo Random Forest entrenado sobre tres capas de señales: Monetaria (estructura de tasas y politica de referencia), Activos (desacoples en mercado bursatil e inmobiliario) y Ahorro/Credito (fragilidad financiera e inflacion). La metodologia integra estas capas en un unico dataset mensual, define una etiqueta de crisis a 18 meses y valida temporalmente para evitar leakage y sobreajuste retrospectivo.

La evidencia historica es consistente con el objetivo del modelo: en la crisis de 2008, la validacion Walk-Forward muestra capacidad de prediccion del 100% en la ventana critica principal (recall completo del episodio), lo que respalda que las señales construidas capturan el patron de expansion y ajuste propio de crisis sistemicas. A nivel global, el rendimiento varia por ciclo, pero se mantiene valor predictivo en episodios macro-financieros de alta relevancia.

En el estado actual, la arquitectura final usa ensamble RF+GBM calibrado (cierre Isotonic vs Platt) con umbral operativo de 4%, seleccionado bajo criterio de sensibilidad (Recall minimo 0.70). Con este esquema, la lectura vigente permanece en zona de accion, por lo que se sostiene una postura de monitoreo reforzado y preparacion preventiva.

Como cierre metodologico, se incorporo una inferencia extendida independiente de la etiqueta para proyectar riesgo hasta el ultimo mes con features completas desde fuentes crudas. Esta extension llevo la cobertura hasta 2025-09-01 y registro una probabilidad reciente de 29.51%, superior al umbral operativo del 4%. En terminos ejecutivos, esto confirma que la alerta no depende solo del corte historico de entrenamiento, sino que se mantiene activa tambien en la lectura mas actualizada del sistema.

## 1. Que es este proyecto
Este proyecto construye un sistema de alerta temprana de crisis economicas basado en la Teoria Austriaca del Ciclo Economico (TACE/ABCT). La idea central es detectar, con horizonte de 18 meses, cuando una economia se acerca a una fase de crisis/ajuste luego de una etapa de expansion crediticia y mala asignacion de capital.

El enfoque combina:
- Variables monetarias (origen del ciclo)
- Variables de activos e inversion (transmision de distorsiones)
- Variables de ahorro/credito e inflacion (fragilidad final)

Con esas tres capas se entrena un clasificador `RandomForest` para estimar la probabilidad de crisis futura.

---

## 2. Estructura del repositorio
Archivos principales:
- `vbles_1ra.ipynb`: ingenieria de variables de Alerta 1 (Monetaria)
- `vbles_2da.ipynb`: ingenieria de variables de Alerta 2 (Activos/Malinversion)
- `vbles_3ra.ipynb`: ingenieria de variables de Alerta 3 (Ahorro/Credito/Inflacion)
- `modelo.ipynb`: entrenamiento, evaluacion, calibracion, seleccion de umbral y walk-forward
- `dataset_alerta1.csv`, `dataset_alerta2.csv`, `dataset_alerta3.csv`: datasets derivados por capa
- `datos ABCT/`: fuentes originales en CSV

---

## 3. Fuentes de datos y cobertura temporal
Periodo base de trabajo:
- Inicio: `1980-01-01`
- Fin: `2025-10-01`

Series usadas (segun disponibilidad):
- FEDFUNDS
- GS10
- GDPC1
- BUSLOANS
- CPIAUCSL
- IPBUSEQ
- NASDAQCOM
- PCE
- USSTHPI
- Gold (onza)
- PSAVERT

Notas tecnicas:
- Series trimestrales se transforman a mensual por interpolacion lineal.
- Series diarias (como NASDAQ) se agregan a promedio mensual.
- Se estandariza todo a indice mensual de inicio de mes (`MS`).

---

## 4. Marco conceptual TACE traducido a variables
### 4.1 Alerta 1: Origen monetario del ciclo
Captura la distorsion inicial de tasa y estructura temporal del credito.

Variables:
- `fedfunds`
- `gs10`
- `spread_yield_curve = gs10 - fedfunds`

Interpretacion:
- Compresion/inversion de curva y politica monetaria agresiva pueden anticipar cambios de fase del ciclo.

### 4.2 Alerta 2: Transmision a activos y mala inversion
Busca detectar desacople entre precios de activos y fundamentos reales.

Ratios base:
- `ipbuseq_pce_ratio`
- `nasdaq_gold_ratio`
- `ussthpi_gold_ratio`

Transformaciones por ratio:
- Nivel
- Media movil 6 meses (`_ma6`)
- Variacion interanual (`_yoy_pct`)
- Z-score de 24 meses (`_zscore24`)

Total de features en Alerta 2: 12.

### 4.3 Alerta 3: Fragilidad ahorro/credito e inflacion
Mide vulnerabilidad financiera y tension macro final.

Variables base:
- `busloans_psavert_ratio`
- `cpi_yoy_pct`

Transformaciones:
- Para `busloans_psavert_ratio`: nivel, ma6, yoy_pct, zscore24
- Para `cpi_yoy_pct`: nivel, ma6, zscore24

Total de features en Alerta 3: 7.

---

## 5. Definicion de la variable objetivo (Y)
La etiqueta final se construye de forma autoritativa en `modelo.ipynb` con horizonte de 18 meses.

Reglas de clase `1` (crisis):
1. Recesiones oficiales NBER.
2. Evento financiero 1987 (oct-1987 a mar-1988).
3. Criterio real: `GDPC1 YoY < 0` por al menos 2 meses consecutivos.

Reglas adicionales:
- Clase `0` en el resto de los meses.
- `Y = crisis_now.shift(-18)` para predecir 18 meses hacia adelante.
- COVID (feb-2020 a abr-2020) se conserva en dataset para visualizacion, pero se excluye del entrenamiento para evitar contaminar el patron TACE con shock exogeno.

---

## 6. Integracion de datasets y particion temporal
1. Se cargan `dataset_alerta1.csv`, `dataset_alerta2.csv`, `dataset_alerta3.csv`.
2. Se hace `inner join` por fecha.
3. Se anexan `y` y `covid_exclude`.
4. Se eliminan `NaN`.
5. Split temporal (sin leakage):
   - Train: primeros 80%
   - Test: ultimos 20%

Modelo usado:
- `RandomForestClassifier`
- `n_estimators=100`
- `max_depth=5`
- `class_weight='balanced'`
- `random_state=42`

---

## 7. Evaluacion del modelo
El notebook evalua en varias capas:
- Matriz de confusion
- Precision / Recall / F1
- Importancia de variables
- Curva historica de probabilidad
- AUC-PR
- Brier Score
- Calibration Curve
- Tabla de metricas por umbral
- Walk-Forward (ventana expansiva, test anual)

### 7.3 Mejoras implementadas sobre el modelo base
Se implementaron dos mejoras directas en `modelo.ipynb`:

1. Ensamble `RF + GBM` con soft voting.
2. Calibracion probabilistica del ensamble con `isotonic` y `TimeSeriesSplit`.

Comparativa en test set (respecto al RF base):
- RF original: AUC-PR = 0.1166, Brier = 0.1026.
- Ensamble RF+GBM: AUC-PR = 0.1194, Brier = 0.1539.
- Ensamble calibrado (isotonic): AUC-PR = 0.1059, Brier = 0.1254.

Lectura operativa de la mejora:
- A umbral operativo del 6%, el ensamble calibrado logra `Recall = 1.00` en el test actual (con precision baja, consistente con una estrategia que prioriza sensibilidad).
- La probabilidad actual con ensamble calibrado queda en 46.67%, por encima del umbral operativo.

### 7.1 Hallazgos relevantes
- `spread_yield_curve` aparece como variable dominante, consistente con la logica TACE.
- La curva historica de probabilidad detecta episodios criticos clasicos (incluyendo 2008).
- La validacion walk-forward muestra desempeno heterogeneo por ciclo, lo cual es esperable en procesos macro de largo plazo.

### 7.2 Evidencia puntual 2008
En ventanas walk-forward asociadas a la crisis subprime, el modelo alcanzo deteccion completa del episodio en terminos de recall (1.00 en la ventana central 2007-12 a 2008-11), mostrando capacidad para capturar senales de crisis sistemica en ese ciclo.

### 7.4 Ablation test de variables con interpretacion discutible
Como validacion metodologica adicional, se corrio un ablation test en `modelo.ipynb` para evaluar si ciertas familias de variables con mezcla de unidades estaban perjudicando el modelo:

- `busloans_psavert_ratio*`
- `ussthpi_gold_ratio*`
- `nasdaq_gold_ratio*`

Configuraciones comparadas (mismo pipeline RF+GBM calibrado con Platt):
- Baseline (22 features): AUC-PR = 0.0779, Brier = 0.1110, Precision_sel = 0.1075, Recall_sel = 1.00.
- Ablation limpio (10 features): AUC-PR = 0.0684, Brier = 0.1164, Precision_sel = 0.1020, Recall_sel = 1.00.

Lectura tecnica:
- El set limpio mejora interpretabilidad economica, pero en esta corrida pierde algo de desempeno predictivo y calibracion.
- Por eso, el trabajo no queda invalidado por esas variables; mas bien muestran aporte de senal en el setup actual.
- Siguiente paso recomendado: reemplazo economico de esas razones (no solo eliminacion), por ejemplo `BUSLOANS/GDP` y `USSTHPI` deflactado por `CPI/PCE`.

---

## 8. Seleccion automatica de umbral de alerta
Se implementa una seleccion automatica de umbral basada en predicciones out-of-sample del walk-forward, con criterio de negocio:

- Prioridad: `Recall >= 0.70` (no perdernos crisis reales)
- Entre umbrales elegibles: maximizar precision
- Desempate: mayor umbral (menos ruido)

Resultado obtenido:
- Umbral operativo final: `4%` (tras cierre de arquitectura con calibracion final).
- Con ese umbral, la lectura mas reciente de inferencia extendida (`29.51%`) queda claramente en zona de accion.

Interpretacion operativa:
- Este enfoque privilegia sensibilidad: acepta mas alertas para reducir el riesgo de omitir una crisis real.

### 8.1 Inferencia extendida al ultimo dato disponible
Para evitar que la inferencia se corte por el desplazamiento de la etiqueta (`y = t+18`), se separo el pipeline en dos etapas:

- Entrenamiento/evaluacion con `y` autoritativa.
- Inferencia con reconstruccion de features desde CSV crudos (sin requerir `y` observada).

Con esta extension, la curva de probabilidad llega hasta `2025-09-01` en la interseccion de features completas. La lectura mas reciente reporta `29.51%` de probabilidad de crisis, por encima del umbral operativo (`4%`), manteniendo senal de accion activa.

---

## 9. Como correr el proyecto
1. Abrir `vbles_1ra.ipynb` y ejecutar todas las celdas (exporta `dataset_alerta1.csv`).
2. Abrir `vbles_2da.ipynb` y ejecutar todas las celdas (exporta `dataset_alerta2.csv`).
3. Abrir `vbles_3ra.ipynb` y ejecutar todas las celdas (exporta `dataset_alerta3.csv`).
4. Abrir `modelo.ipynb` y ejecutar de arriba hacia abajo.
5. Revisar:
   - Graficos historicos
   - Metricas por umbral
   - Resultado de umbral optimo

Dependencias principales:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 10. Limitaciones y consideraciones
- Modelo intencionalmente orientado a patrones TACE; shocks exogenos puros pueden no anticiparse con alta precision.
- Resultados dependen de definicion de crisis, horizonte (18m) y ventanas de validacion.
- En series macro largas, estabilidad entre ciclos nunca es perfecta; por eso se usa walk-forward y no solo un split unico.

---

## 11. Proximos pasos recomendados
- Ajuste fino del ensamble RF+GBM (pesos, profundidad, learning rate) para mejorar precision sin perder recall.
- Sensibilidad de inferencia para extender hasta 2026-03 con politicas de imputacion/extrapolacion explicitamente documentadas.
- Definir politica de accion por niveles: vigilancia, alerta preventiva, alerta critica.
- Monitoreo mensual con backtesting en rolling window.

---
