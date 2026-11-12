import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

st.set_page_config(page_title="Riesgo de salud (prototipo académico)", page_icon="🩺", layout="wide")

ORDEN = ["Bajo", "Medio", "Alto"]
COLORES = {"Bajo": "#59A14F", "Medio": "#F2B134", "Alto": "#E15759"}
INTEGRANTES = ["Roberto Andrés Cortes Castillejo", "Roiner Luis Mejía Arrieta", "Santiago De Jesús Remolina López"]
NUMERICAS = ["Edad", "IMC", "Glucosa_mg_dL", "Presion_Sistolica_mmHg", "Colesterol_mg_dL",
             "Horas_Sueno", "Nivel_Estres", "Consultas_Medicas", "Mes_Numero"]
CATEGORICAS = ["Genero", "Actividad_Fisica", "Fumador"]
PREDICTORES = NUMERICAS + CATEGORICAS
CATEGORIAS = {"Genero": ["Femenino", "Masculino", "Otro/No reporta"],
              "Actividad_Fisica": ["Baja", "Moderada", "Alta"],
              "Fumador": ["No", "Sí"]}
ETIQUETAS = {"Edad": "Edad (años)", "IMC": "IMC (kg/m²)", "Glucosa_mg_dL": "Glucosa (mg/dL)",
             "Presion_Sistolica_mmHg": "Presión sistólica (mmHg)", "Colesterol_mg_dL": "Colesterol total (mg/dL)",
             "Horas_Sueno": "Horas de sueño por noche", "Nivel_Estres": "Nivel de estrés (1 a 10)",
             "Consultas_Medicas": "Consultas médicas en 12 meses", "Mes_Numero": "Mes de registro (1 a 12)",
             "Genero": "Género", "Actividad_Fisica": "Actividad física", "Fumador": "Fumador"}


@st.cache_resource
def cargar_modelo():
    return joblib.load("pipeline_riesgo.joblib")  # se carga, nunca se reentrena


@st.cache_data
def cargar_datos():
    return pd.read_csv("datos_limpios.csv"), pd.read_csv("datos_prueba.csv")


modelo = cargar_modelo()
datos, prueba = cargar_datos()
RANGOS = {c: (float(np.floor(datos[c].min())), float(np.ceil(datos[c].max()))) for c in NUMERICAS}


def validar(e):
    """Valida tipos, rangos (los vistos en el entrenamiento) y categorías."""
    errores = []
    for c in NUMERICAS:
        v = e[c]
        if v is None or not isinstance(v, (int, float, np.integer, np.floating)) or np.isnan(v):
            errores.append(f"{ETIQUETAS[c]}: debe ser un número.")
        elif not (RANGOS[c][0] <= v <= RANGOS[c][1]):
            errores.append(f"{ETIQUETAS[c]}: {v} está fuera del rango usado para entrenar el modelo "
                           f"({RANGOS[c][0]:g} a {RANGOS[c][1]:g}).")
    for c in CATEGORICAS:
        if e[c] not in CATEGORIAS[c]:
            errores.append(f"{ETIQUETAS[c]}: categoría no válida.")
    return errores


def predecir(fila_df):
    probs = modelo.predict_proba(fila_df)[0]
    por_clase = dict(zip(modelo.classes_, probs))  # classes_ está en orden alfabético
    return modelo.predict(fila_df)[0], por_clase


st.title("🩺 Clasificación del nivel de riesgo de salud")
st.warning("Prototipo académico con datos sintéticos. No es un diagnóstico y no sustituye una decisión profesional.")

tab1, tab2, tab3, tab4 = st.tabs(["Proyecto", "Exploración de datos", "Predicción", "Métricas y limitaciones"])

# ---------------- Proyecto ----------------
with tab1:
    st.header("Presentación del proyecto")
    st.markdown(
        "**Objetivo:** predecir el nivel de riesgo (Bajo, Medio o Alto) a partir de datos demográficos, "
        "mediciones clínicas simuladas y hábitos, con especial atención a los errores sobre la clase Alto.\n\n"
        "**Modelo:** regresión logística multinomial con preprocesamiento (imputación por mediana, "
        "estandarización y codificación one-hot) guardado como un único pipeline.\n\n"
        "**Fuente de los datos:** conjunto sintético generado con fines docentes "
        "(`Dataset_Sintetico_Salud_EDA.xlsx`, recurso de la asignatura Minería de Datos), 497 registros "
        "únicos de enero a diciembre de 2025.")
    st.subheader("Integrantes")
    for n in INTEGRANTES:
        st.write(f"- {n}")
    st.caption("Universidad de la Costa · Minería de Datos · Docente: José Escorcia Gutierrez")

# ---------------- EDA ----------------
with tab2:
    st.header("Visualizaciones del análisis exploratorio")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Distribución del riesgo")
        conteo = datos["Nivel_Riesgo"].value_counts().reindex(ORDEN)
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.bar(ORDEN, conteo.values, color=[COLORES[o] for o in ORDEN])
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 2, f"{v} ({v / conteo.sum():.1%})", ha="center", fontsize=8)
        ax.set(ylabel="Registros", xlabel="Nivel de riesgo")
        st.pyplot(fig)
    with c2:
        st.subheader("Glucosa según riesgo")
        fig, ax = plt.subplots(figsize=(4, 3.4))
        sns.boxplot(data=datos, x="Nivel_Riesgo", y="Glucosa_mg_dL", order=ORDEN, hue="Nivel_Riesgo", palette=COLORES, legend=False, ax=ax)
        ax.set(xlabel="Nivel de riesgo", ylabel="Glucosa (mg/dL)")
        st.pyplot(fig)
    with c3:
        st.subheader("Riesgo por actividad física")
        ct = pd.crosstab(datos["Actividad_Fisica"], datos["Nivel_Riesgo"]).reindex(CATEGORIAS["Actividad_Fisica"])[ORDEN]
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(4, 3.4))
        pct.plot(kind="bar", stacked=True, color=[COLORES[o] for o in ORDEN], ax=ax, rot=0)
        ax.set(xlabel="Actividad física", ylabel="% de registros", ylim=(0, 100))
        ax.legend(title="Riesgo", fontsize=7)
        st.pyplot(fig)
    st.caption("Las asociaciones describen los datos sintéticos y no implican causalidad.")

# ---------------- Predicción ----------------
with tab3:
    st.header("Consultar una predicción")
    with st.form("formulario"):
        a, b, c = st.columns(3)
        with a:
            edad = st.number_input(ETIQUETAS["Edad"], 0, 120, 45, 1)
            imc = st.number_input(ETIQUETAS["IMC"], 5.0, 100.0, 26.8, 0.1)
            glucosa = st.number_input(ETIQUETAS["Glucosa_mg_dL"], 20.0, 600.0, 103.0, 1.0)
            presion = st.number_input(ETIQUETAS["Presion_Sistolica_mmHg"], 50, 300, 122, 1)
        with b:
            colesterol = st.number_input(ETIQUETAS["Colesterol_mg_dL"], 50.0, 500.0, 190.0, 1.0)
            sueno = st.number_input(ETIQUETAS["Horas_Sueno"], 0.0, 24.0, 7.0, 0.1)
            estres = st.slider(ETIQUETAS["Nivel_Estres"], 1, 10, 5)
            consultas = st.number_input(ETIQUETAS["Consultas_Medicas"], 0, 50, 3, 1)
        with c:
            mes = st.selectbox(ETIQUETAS["Mes_Numero"], list(range(1, 13)), index=5)
            genero = st.selectbox(ETIQUETAS["Genero"], CATEGORIAS["Genero"])
            actividad = st.selectbox(ETIQUETAS["Actividad_Fisica"], CATEGORIAS["Actividad_Fisica"])
            fumador = st.selectbox(ETIQUETAS["Fumador"], CATEGORIAS["Fumador"])
        enviar = st.form_submit_button("Predecir")

    if enviar:
        entrada = {"Edad": edad, "IMC": imc, "Glucosa_mg_dL": glucosa, "Presion_Sistolica_mmHg": presion,
                   "Colesterol_mg_dL": colesterol, "Horas_Sueno": sueno, "Nivel_Estres": estres,
                   "Consultas_Medicas": consultas, "Mes_Numero": mes, "Genero": genero,
                   "Actividad_Fisica": actividad, "Fumador": fumador}
        errores = validar(entrada)
        if errores:
            for e in errores:
                st.error(e)
        else:
            clase, probs = predecir(pd.DataFrame([entrada])[PREDICTORES])
            st.success(f"Nivel de riesgo predicho: **{clase}** (probabilidad {probs[clase]:.2%})")
            st.bar_chart(pd.Series({o: probs[o] for o in ORDEN}, name="Probabilidad"))

    with st.expander("Verificar con un registro conocido del conjunto de prueba"):
        idx = st.number_input("Posición del registro (0 a %d)" % (len(prueba) - 1), 0, len(prueba) - 1, 0, 1)
        fila = prueba.iloc[[int(idx)]]
        st.dataframe(fila[PREDICTORES + ["Nivel_Riesgo"]], hide_index=True)
        clase, probs = predecir(fila[PREDICTORES])
        st.write(f"Predicción: **{clase}** · Real: **{fila['Nivel_Riesgo'].iloc[0]}** · "
                 + " · ".join(f"{o}: {probs[o]:.4f}" for o in ORDEN))

# ---------------- Métricas ----------------
with tab4:
    st.header("Métricas finales en el conjunto de prueba")
    y_real = prueba["Nivel_Riesgo"]
    y_pred = modelo.predict(prueba[PREDICTORES])
    p, r, f, s = precision_recall_fscore_support(y_real, y_pred, labels=ORDEN)
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{accuracy_score(y_real, y_pred):.3f}")
    m2.metric("F1 macro", f"{f1_score(y_real, y_pred, average='macro'):.3f}")
    m3.metric("Recall de Alto", f"{r[2]:.3f}")
    st.dataframe(pd.DataFrame({"Clase": ORDEN, "Precision": p.round(3), "Recall": r.round(3),
                               "F1": f.round(3), "Registros": s}), hide_index=True)
    st.caption("Validación cruzada de 5 particiones sobre el entrenamiento (informe del Avance 2): "
               "accuracy 0.844 ± 0.045, F1 macro 0.849 ± 0.045.")
    st.subheader("Matriz de confusión")
    cm = confusion_matrix(y_real, y_pred, labels=ORDEN)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ORDEN, yticklabels=ORDEN, cbar=False, ax=ax)
    ax.set(xlabel="Clase predicha", ylabel="Clase real")
    st.pyplot(fig)
    st.header("Limitaciones")
    st.markdown(
        "- Los datos son **sintéticos** y la etiqueta se construyó a partir de otras variables del mismo conjunto, "
        "por lo que el rendimiento refleja en parte la recuperación de esa regla y no es generalizable.\n"
        "- La prueba es pequeña (125 registros, 33 de la clase Alto): el recall de Alto tiene un intervalo amplio.\n"
        "- Se usó una sola partición y no se optimizaron hiperparámetros.\n"
        "- La categoría «Otro/No reporta» tiene solo 12 registros.\n"
        "- Prototipo académico: no es un diagnóstico clínico.")
