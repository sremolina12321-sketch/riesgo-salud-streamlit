# riesgo-salud-streamlit
# Clasificación del nivel de riesgo de salud con Streamlit

Prototipo académico con datos sintéticos. No es un diagnóstico y no sustituye una decisión profesional.

Aplicación pública:https://riesgo-salud-app-h2ma5ofnu2j2a5w2oqfk9d.streamlit.app/

## Integrantes
- Roberto Andrés Cortes Castillejo
- Roiner Luis Mejía Arrieta
- Santiago De Jesús Remolina López

Universidad de la Costa · Minería de Datos · Docente: José Escorcia Gutierrez · Tercer corte, Actividad 1

## Objetivo
Predecir el nivel de riesgo (Bajo, Medio o Alto) a partir de datos demográficos, mediciones clínicas simuladas y hábitos, con especial atención a los errores sobre la clase Alto.

## Datos
Conjunto sintético generado con fines docentes (`Dataset_Sintetico_Salud_EDA.xlsx`, recurso de la asignatura). Tras eliminar 3 duplicados exactos quedan 497 registros. La etiqueta `Nivel_Riesgo` se construyó a partir de otras variables del mismo conjunto.

## Modelo
Regresión logística multinomial (pesos de clase balanceados) dentro de un único pipeline: imputación por mediana y estandarización para las variables numéricas; imputación por moda y codificación one-hot para las categóricas. Partición 75 %/25 % estratificada, semilla 42. El modelo del segundo corte **no se modificó**; la app carga el pipeline guardado con `joblib` y no reentrena.

## Resultados en el conjunto de prueba (125 registros)
| Métrica | Valor |
|---|---|
| Accuracy | 0.856 |
| F1 macro | 0.859 |
| Recall de Alto | 0.909 (30 de 33) |
| Validación cruzada (5 particiones), F1 macro | 0.849 ± 0.045 |

Matriz de confusión (filas: real; columnas: predicho Bajo, Medio, Alto):
Bajo 37-6-0 · Medio 3-40-6 · Alto 0-3-30.

## Limitaciones
- Datos sintéticos y etiqueta derivada de otras variables: el rendimiento no es generalizable.
- Prueba pequeña (33 registros de Alto): el recall de Alto tiene un intervalo amplio.
- Una sola partición y sin optimización de hiperparámetros.
- La categoría «Otro/No reporta» tiene solo 12 registros.

## Archivos
- `app.py`: aplicación Streamlit.
- `pipeline_riesgo.joblib`: pipeline completo (preprocesamiento + modelo).
- `datos_limpios.csv`, `datos_prueba.csv`: datos para las gráficas, las métricas y la verificación.
- `requirements.txt`: dependencias con las versiones usadas para guardar el modelo.
- `Avance_2_Proyecto_Mineria_Datos.ipynb`: notebook con el EDA, el modelado y la exportación del pipeline (sección 9).

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pruebas realizadas
1. Caso normal: registro de referencia → Medio, probabilidad 86.55 %.
2. Entrada inválida: edad 110 → mensaje de error y sin predicción.
3. Registro conocido del conjunto de prueba (posición 0): predicción Medio, real Medio, con las mismas probabilidades que el notebook (Alto 0.1268, Bajo 0.0077, Medio 0.8655).
