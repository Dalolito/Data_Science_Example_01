"""
Dashboard de prueba - Estadísticas Mundial (datos sintéticos)
Ejecutar con: streamlit run main_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Mundial (datos sintéticos)",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Dashboard de prueba - Mundial (datos 100% sintéticos)")
st.caption(
    "Todos los datos de este tablero se generan aleatoriamente al vuelo con fines "
    "de demostración técnica. No representan estadísticas reales de ningún equipo, "
    "jugador ni organización."
)

# --------------------------------------------------------------------------
# GENERACIÓN DE DATOS SINTÉTICOS (200 registros, 8 columnas, tipos mixtos)
# --------------------------------------------------------------------------
@st.cache_data
def generar_datos(semilla: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    equipos = [
        "Argentina", "Brasil", "Francia", "Alemania", "España",
        "Inglaterra", "Portugal", "Países Bajos", "Uruguay", "Colombia",
        "Croacia", "Marruecos", "Japón", "Estados Unidos", "México",
    ]
    confederaciones = {
        "Argentina": "CONMEBOL", "Brasil": "CONMEBOL", "Uruguay": "CONMEBOL", "Colombia": "CONMEBOL",
        "Francia": "UEFA", "Alemania": "UEFA", "España": "UEFA", "Inglaterra": "UEFA",
        "Portugal": "UEFA", "Países Bajos": "UEFA", "Croacia": "UEFA",
        "Marruecos": "CAF", "Japón": "AFC", "Estados Unidos": "CONCACAF", "México": "CONCACAF",
    }
    fases = ["Grupos", "Octavos", "Cuartos", "Semifinal", "Final"]

    equipo_col = rng.choice(equipos, size=n)

    df = pd.DataFrame({
        "equipo": equipo_col,                                                   # categórica (texto)
        "confederacion": [confederaciones[e] for e in equipo_col],              # categórica (texto)
        "fase": rng.choice(fases, size=n, p=[0.45, 0.25, 0.15, 0.10, 0.05]),      # categórica ordinal
        "goles_favor": rng.poisson(1.6, size=n),                                # entero (discreta)
        "goles_contra": rng.poisson(1.1, size=n),                               # entero (discreta)
        "posesion_pct": np.clip(rng.normal(50, 10, size=n), 20, 85).round(1),    # float continua
        "fecha_partido": pd.to_datetime("2026-06-11") + pd.to_timedelta(
            rng.integers(0, 30, size=n), unit="D"
        ),                                                                       # fecha
        "gano_partido": rng.choice([True, False], size=n, p=[0.5, 0.5]),         # booleana
    })

    # ---- Variable ilustrativa: "índice de decisiones arbitrales favorables" ----
    # Se genera de forma ALEATORIA para todos los equipos. Para fines puramente
    # demostrativos (mostrar cómo se vería un sesgo en un gráfico), se le añade
    # un pequeño desplazamiento artificial a Argentina. Esto es un dato FICTICIO
    # de prueba, no una afirmación real sobre la FIFA ni sobre ningún equipo.
    base_idx = rng.normal(50, 12, size=n)
    ajuste_demo = np.where(df["equipo"] == "Argentina", 8, 0)  # sesgo artificial de demo
    df["indice_arbitral_sintetico"] = np.clip(base_idx + ajuste_demo, 0, 100).round(1)

    return df


with st.sidebar:
    st.header("⚙️ Controles de simulación")
    semilla = st.number_input("Semilla aleatoria", min_value=0, max_value=9999, value=42)
    if st.button("🔄 Regenerar datos sintéticos"):
        st.cache_data.clear()
    st.markdown("---")

df = generar_datos(semilla)

st.info(
    "ℹ️ La columna **indice_arbitral_sintetico** es un dato simulado creado solo para "
    "ilustrar visualmente cómo se vería un posible sesgo en un gráfico. Se generó "
    "aleatoriamente y no proviene de ninguna fuente real ni constituye una afirmación "
    "verídica sobre la FIFA o algún equipo.",
    icon="⚠️",
)

with st.expander("📄 Ver muestra de datos crudos (200 registros x 8 columnas)"):
    st.dataframe(df, use_container_width=True, height=300)
    st.caption(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]} | Tipos: {dict(df.dtypes.astype(str))}")

st.markdown("---")

# --------------------------------------------------------------------------
# ESQUEMA DE MÉTRICAS
# --------------------------------------------------------------------------
tab_cuanti, tab_cuali, tab_graf, tab_umbral = st.tabs(
    ["📊 Estadística cuantitativa", "🗂️ Estadística cualitativa", "📈 Análisis gráfico", "🎯 Umbrales interactivos"]
)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

# ---- TAB 1: CUANTITATIVA ----
with tab_cuanti:
    st.subheader("Resumen estadístico de variables numéricas")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Goles a favor (media)", f"{df['goles_favor'].mean():.2f}")
    c2.metric("Goles en contra (media)", f"{df['goles_contra'].mean():.2f}")
    c3.metric("Posesión promedio %", f"{df['posesion_pct'].mean():.1f}")
    c4.metric("Desv. estándar posesión", f"{df['posesion_pct'].std():.1f}")

    st.dataframe(df[num_cols].describe().T, use_container_width=True)

    st.subheader("Matriz de correlación")
    corr = df[num_cols].corr(numeric_only=True)
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlación entre variables numéricas",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ---- TAB 2: CUALITATIVA ----
with tab_cuali:
    st.subheader("Distribución de variables categóricas")
    col_cat = st.selectbox("Selecciona una variable categórica", cat_cols, index=0)
    conteo = df[col_cat].value_counts().reset_index()
    conteo.columns = [col_cat, "frecuencia"]
    conteo["porcentaje"] = (conteo["frecuencia"] / conteo["frecuencia"].sum() * 100).round(1)
    st.dataframe(conteo, use_container_width=True)

    fig_pie = px.pie(conteo, names=col_cat, values="frecuencia", title=f"Distribución de {col_cat}", hole=0.35)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(f"**Moda de `{col_cat}`:** {df[col_cat].mode().iloc[0]}")

# ---- TAB 3: ANÁLISIS GRÁFICO DINÁMICO ----
with tab_graf:
    st.subheader("Explorador dinámico de variables")
    colA, colB, colC = st.columns(3)
    with colA:
        var_x = st.selectbox("Variable eje X", df.columns, index=df.columns.get_loc("equipo"))
    with colB:
        var_y = st.selectbox("Variable eje Y", num_cols, index=num_cols.index("goles_favor"))
    with colC:
        tipo_grafico = st.selectbox(
            "Tipo de gráfico", ["Barras", "Dispersión", "Caja (Box)", "Histograma", "Línea de tiempo"]
        )

    color_por = st.selectbox("Colorear por", ["(ninguno)"] + cat_cols, index=0)
    color_arg = None if color_por == "(ninguno)" else color_por

    if tipo_grafico == "Barras":
        agg = df.groupby(var_x, as_index=False)[var_y].mean()
        fig = px.bar(agg, x=var_x, y=var_y, title=f"{var_y} promedio por {var_x}", color=var_x)
    elif tipo_grafico == "Dispersión":
        fig = px.scatter(df, x=var_x, y=var_y, color=color_arg, title=f"{var_y} vs {var_x}")
    elif tipo_grafico == "Caja (Box)":
        fig = px.box(df, x=var_x, y=var_y, color=color_arg, title=f"Distribución de {var_y} por {var_x}")
    elif tipo_grafico == "Histograma":
        fig = px.histogram(df, x=var_y, color=color_arg, nbins=30, title=f"Histograma de {var_y}")
    else:  # Línea de tiempo
        serie = df.groupby("fecha_partido", as_index=False)[var_y].mean()
        fig = px.line(serie, x="fecha_partido", y=var_y, markers=True, title=f"{var_y} promedio en el tiempo")

    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 4: UMBRALES INTERACTIVOS ----
with tab_umbral:
    st.subheader("Gráfico de barras con línea de umbral")
    var_umbral = st.selectbox("Variable numérica para evaluar", num_cols, index=num_cols.index("indice_arbitral_sintetico"))
    agrupar_por = st.selectbox("Agrupar por", cat_cols, index=cat_cols.index("equipo"))
    umbral = st.slider(
        "Valor de umbral",
        float(df[var_umbral].min()), float(df[var_umbral].max()),
        float(df[var_umbral].mean()),
    )

    agg = df.groupby(agrupar_por, as_index=False)[var_umbral].mean().sort_values(var_umbral, ascending=False)
    agg["supera_umbral"] = agg[var_umbral] > umbral

    fig_umbral = px.bar(
        agg, x=agrupar_por, y=var_umbral, color="supera_umbral",
        color_discrete_map={True: "crimson", False: "steelblue"},
        title=f"{var_umbral} promedio por {agrupar_por} (umbral = {umbral:.1f})",
    )
    fig_umbral.add_hline(y=umbral, line_dash="dash", line_color="black", annotation_text="Umbral")
    st.plotly_chart(fig_umbral, use_container_width=True)

    n_supera = int(agg["supera_umbral"].sum())
    st.metric(f"Categorías de '{agrupar_por}' que superan el umbral", n_supera)

    if var_umbral == "indice_arbitral_sintetico":
        st.caption(
            "Recordatorio: este índice es completamente sintético y se generó con un "
            "desplazamiento artificial para fines de demostración visual, no refleja "
            "datos reales de arbitraje ni decisiones de la FIFA."
        )

st.markdown("---")
st.caption("Dashboard de prueba generado con Streamlit + Plotly. Datos 100% sintéticos y regenerables desde la barra lateral.")

