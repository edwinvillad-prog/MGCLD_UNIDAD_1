# ===============================================================
# 🧮 Laboratorio Interactivo — ANOVA de un Factor y Factorial (Unidad 4)
# Autor: M.Sc. Edwin Villarreal, Fís. — Universidad Politécnica Salesiana (UPS)
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# ---------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------
st.set_page_config(page_title="Laboratorio ANOVA — Unidad 4", page_icon="🧮", layout="wide")

UPS_BLUE = "#002F6C"
UPS_GOLD = "#F7B500"
UPS_TEXT = "#1E1E1E"

st.markdown(f"""
<h1 style='text-align:center; color:{UPS_BLUE};'>
🧮 Laboratorio Interactivo — ANOVA de un Factor y Factorial
</h1>
<h4 style='text-align:center; color:black;'>
Autor: M.Sc. Edwin Villarreal, Fís. — Universidad Politécnica Salesiana
</h4>
""", unsafe_allow_html=True)

st.markdown("""
Este laboratorio permite analizar los **efectos de uno o más factores categóricos**
sobre una **variable de respuesta cuantitativa**, aplicando métodos de **ANOVA** de un factor y factorial.
""")

# ---------------------------------------------------------------
# CREACIÓN DE PESTAÑAS
# ---------------------------------------------------------------
tabs = st.tabs([
    "📂 Datos",
    "🧮 ANOVA de un Factor",
    "⚙️ ANOVA Factorial",
    "📊 Supuestos y Conclusiones"
])

# ---------------------------------------------------------------
# TAB 1 — CARGA DE DATOS
# ---------------------------------------------------------------
with tabs[0]:
    st.header("📂 Carga de datos")

    archivo = st.file_uploader("Suba un archivo CSV o Excel", type=["csv", "xlsx"])

    if archivo:
        try:
            if archivo.name.endswith(".csv"):
                df = pd.read_csv(archivo)
            else:
                df = pd.read_excel(archivo)

            st.session_state["df"] = df
            st.success("✅ Datos cargados correctamente.")
            st.dataframe(df.head())

            st.markdown("""
            <div style="background-color:#F8F9FA; border-left:4px solid #002F6C; padding:10px; border-radius:8px;">
            <b>Observaciones:</b><br>
            • La correcta selección y carga de datos garantiza la validez del análisis estadístico.<br>
            • Asegúrate de que las variables categóricas estén correctamente codificadas (por ejemplo, <b>Factor, Tratamiento, Día</b>).<br>
            • Los datos deben contener al menos una variable dependiente cuantitativa y una o más variables categóricas.<br>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")
    else:
        st.info("Carga un archivo con tus datos para comenzar el análisis.")
# ===============================================================
# 🧮 TAB 2 — ANOVA DE UN FACTOR (CON TUKEY Y CONCLUSIONES + FIX)
# ===============================================================
with tabs[1]:
    st.header("🧮 ANOVA de un factor")

    # -----------------------------------------------------------
    # Verificación de datos cargados
    # -----------------------------------------------------------
    if "df" not in st.session_state:
        st.warning("Primero carga un archivo en la pestaña 📂 Datos.")
    else:
        df = st.session_state["df"]

        numericas = df.select_dtypes(include=np.number).columns.tolist()
        categoricas = df.select_dtypes(exclude=np.number).columns.tolist()

        y = st.selectbox("Variable dependiente (numérica):", numericas)
        factor = st.selectbox("Factor (categórico):", categoricas)

        # ===========================================================
        # EJECUCIÓN COMPLETA DENTRO DEL BOTÓN
        # ===========================================================
        if st.button("Calcular ANOVA de un factor"):
            tukey_df = None

            try:
                # ------------------------------------------------------
                # 🔹 1. ANOVA GLOBAL
                # ------------------------------------------------------
                modelo = ols(f"{y} ~ C({factor})", data=df).fit()
                anova_tabla = sm.stats.anova_lm(modelo, typ=2)
                st.session_state["model"] = modelo

                with st.expander("📊 Tabla ANOVA (criterio global de diferencias)", expanded=True):
                    st.dataframe(anova_tabla.round(4), use_container_width=True)
                    st.caption(
                        "El ANOVA indica si existen diferencias globales entre tratamientos. "
                        "Para determinar cuál tratamiento es mejor, se usan las medias y la prueba de Tukey."
                    )

                import plotly.express as px
                fig = px.box(
                    df,
                    x=factor,
                    y=y,
                    color=factor,
                    title=f"Distribución de {y} según {factor}",
                    points="all",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(
                    title_font=dict(size=18, color="#002F6C", family="Arial Black"),
                    xaxis_title=factor,
                    yaxis_title=y,
                    template="simple_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                # ------------------------------------------------------
                # 🔹 2. PRUEBA DE TUKEY
                # ------------------------------------------------------
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                st.subheader("🔍 Comparaciones múltiples (Tukey HSD)")

                try:
                    tukey = pairwise_tukeyhsd(endog=df[y], groups=df[factor], alpha=0.05)
                    tukey_df = pd.DataFrame(
                        data=tukey._results_table.data[1:],
                        columns=tukey._results_table.data[0]
                    )
                    st.dataframe(tukey_df, use_container_width=True)

                    import plotly.graph_objects as go
                    comp_labels = [f"{a} vs {b}" for a, b in zip(tukey_df["group1"], tukey_df["group2"])]
                    colors = ['#002F6C' if bool(sig) else '#A9A9A9' for sig in tukey_df["reject"]]
                    cd = np.column_stack([tukey_df["p-adj"], tukey_df["lower"], tukey_df["upper"]])

                    tukey_fig = go.Figure()
                    tukey_fig.add_trace(go.Bar(
                        x=tukey_df["meandiff"],
                        y=comp_labels,
                        orientation='h',
                        marker_color=colors,
                        hovertemplate=(
                            "Comparación: %{y}<br>"
                            "Diferencia media: %{x:.4f}<br>"
                            "p-ajustada: %{customdata[0]:.4f}<br>"
                            "IC 95%: [%{customdata[1]:.4f}, %{customdata[2]:.4f}]"
                        ),
                        customdata=cd
                    ))
                    tukey_fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#444")
                    tukey_fig.update_layout(
                        title=f"Resultados Tukey HSD — {factor}",
                        title_font=dict(size=17, color="#002F6C", family="Arial Black"),
                        xaxis_title="Diferencia de medias",
                        yaxis_title="Comparaciones",
                        template="simple_white",
                        bargap=0.25
                    )
                    st.plotly_chart(tukey_fig, use_container_width=True)

                    # --------------------------------------------------
                    # 🔹 3. CONCLUSIONES DEL ANOVA Y TUKEY
                    # --------------------------------------------------
                    st.markdown("### 📘 Conclusión")
                    sig = tukey_df["reject"].sum()
                    if sig > 0:
                        st.info(f"""
                        **Conclusión técnica:** Se detectaron {sig} comparaciones significativas (p < 0.05) 
                        entre los niveles del factor **{factor}**. Esto indica que existen diferencias reales 
                        en la variable **{y}** según los tratamientos evaluados.
                        
                        **Conclusión aplicada:** Algunos tratamientos producen valores distintos en la variable 
                        analizada, sugiriendo un efecto del factor experimental sobre el proceso evaluado.
                        """)
                    else:
                        st.info(f"""
                        **Conclusión técnica:** No se encontraron diferencias significativas (p ≥ 0.05) entre los niveles del factor **{factor}**.

                        **Conclusión aplicada:** Los tratamientos presentan un comportamiento homogéneo respecto a **{y}**, 
                        sin evidencia de diferencias atribuibles al factor estudiado.
                        """)

                except Exception as e:
                    st.warning("ℹ️ La prueba de Tukey solo se aplica a un factor con más de dos niveles.")
                    st.error(f"Detalles: {e}")

                # ------------------------------------------------------
                # 🏆 Mejor tratamiento — tabla + gráfico en una fila (versión final)
                # ------------------------------------------------------
                st.markdown("### 🏆 Mejor tratamiento")

                try:
                    # 1️⃣ Calcular medias ordenadas
                    medias = df.groupby(factor)[y].mean().sort_values(ascending=False)
                    medias_df = medias.rename("Media").round(4).reset_index()
                    top = medias_df.iloc[0, 0]  # Se selecciona el tratamiento con mayor media por defecto

                    # 2️⃣ Verificar diferencias con Tukey
                    pmap, sigmap = {}, {}
                    if tukey_df is not None and len(tukey_df) > 0:
                        for _, r in tukey_df.iterrows():
                            g1, g2 = str(r["group1"]), str(r["group2"])
                            p = float(r["p-adj"])
                            if g1 == str(top):
                                pmap[g2] = p
                            elif g2 == str(top):
                                pmap[g1] = p
                        for lvl in medias.index:
                            sigmap[lvl] = True if lvl == top else (pmap.get(lvl, np.nan) < 0.05)
                    else:
                        for lvl in medias.index:
                            sigmap[lvl] = (lvl == top)
                        st.warning("ℹ️ Tukey no disponible (pocos niveles o error previo).")


                    # ===============================================================
                    # 🔹 Gráfico interactivo de medias
                    # ===============================================================
                    import plotly.graph_objects as go
                    niveles = list(medias.index)
                    medias_vals = medias.values
                    colores = ['#003A70' if sigmap.get(niv, False) else '#B0B0B0' for niv in niveles]

                    hover_text = []
                    for niv in niveles:
                        if niv == top:
                            hover_text.append(f"{niv}<br>Media: {medias[niv]:.4f}<br>Comparación base")
                        else:
                            ptxt = pmap.get(niv, np.nan)
                            if np.isnan(ptxt):
                                hover_text.append(f"{niv}<br>Media: {medias[niv]:.4f}<br>p (vs {top}): —")
                            else:
                                hover_text.append(f"{niv}<br>Media: {medias[niv]:.4f}<br>p (vs {top}): {ptxt:.4f}")

                    bar_fig = go.Figure()
                    bar_fig.add_trace(go.Bar(
                        x=niveles,
                        y=medias_vals,
                        marker_color=colores,
                        hovertext=hover_text,
                        hoverinfo="text"
                    ))
                    bar_fig.update_layout(
                        title=f"Medias de {y} por {factor} (referencia: {top})",
                        title_font=dict(size=18, color="#002F6C", family="Arial Black"),
                        xaxis_title=factor,
                        yaxis_title=f"Media de {y}",
                        template="simple_white",
                        bargap=0.25
                    )

                    col1, col2 = st.columns([1.2, 2.3])
                    with col1:
                        st.dataframe(medias_df, use_container_width=True, height=280)
                    with col2:
                        st.plotly_chart(bar_fig, use_container_width=True)

                    # ===============================================================
                    # 🔹 Conclusión aplicada
                    # ===============================================================
                    comparables = [niv for niv in niveles if niv != top]
                    sig_vs_top = [niv for niv in comparables if pmap.get(niv, np.nan) < 0.05]
                    n_comp, n_sig = len(comparables), len(sig_vs_top)

                    if n_comp == 0:
                        texto_conclusion = f"""
                        <div style="background-color:#F8F9FA; border-left:4px solid #002F6C; padding:12px; border-radius:10px;">
                        <b>📘 Conclusión aplicada:</b><br>
                        Solo existe un nivel para el factor <b>{factor}</b>, por lo que no es posible establecer superioridad relativa.
                        </div>
                        """
                    elif n_sig == n_comp:
                        texto_conclusion = f"""
                        <div style="background-color:#F8F9FA; border-left:4px solid #002F6C; padding:12px; border-radius:10px;">
                        <b>📘 Conclusión aplicada:</b><br>
                        El tratamiento <b>{top}</b> presenta la <b>media más alta</b> de <b>{y}</b> y difiere significativamente (Tukey, p &lt; 0.05)
                        de <b>todos</b> los demás niveles de <b>{factor}</b>. Bajo las condiciones evaluadas, puede considerarse el <b>mejor</b>.
                        </div>
                        """
                    elif n_sig > 0:
                        lista_diff = ", ".join(sig_vs_top)
                        texto_conclusion = f"""
                        <div style="background-color:#F8F9FA; border-left:4px solid #002F6C; padding:12px; border-radius:10px;">
                        <b>📘 Conclusión aplicada:</b><br>
                        El tratamiento <b>{top}</b> muestra la <b>media más alta</b> de <b>{y}</b> y difiere significativamente (Tukey, p &lt; 0.05)
                        de: <b>{lista_diff}</b>. Frente a los demás niveles no se evidencia diferencia significativa, por lo que su superioridad
                        es <b>parcial</b> bajo las condiciones evaluadas.
                        </div>
                        """
                    else:
                        texto_conclusion = f"""
                        <div style="background-color:#F8F9FA; border-left:4px solid #002F6C; padding:12px; border-radius:10px;">
                        <b>📘 Conclusión aplicada:</b><br>
                        El tratamiento <b>{top}</b> tiene la media más alta de <b>{y}</b>, pero <b>no</b> difiere significativamente
                        de los otros niveles del factor <b>{factor}</b> (Tukey, p ≥ 0.05). No hay evidencia estadística suficiente
                        para afirmar superioridad bajo las condiciones evaluadas.
                        </div>
                        """
                    st.markdown(texto_conclusion, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error en la sección 'Mejor tratamiento': {e}")

            except Exception as e:
                st.error(f"❌ Error general en el bloque de ANOVA: {e}")


# ===============================================================
# ⚙️ TAB 3 — ANOVA FACTORIAL (dos o más factores) + TUKEY
# ===============================================================
with tabs[2]:
    st.header("⚙️ ANOVA factorial (dos o más factores)")

    if "df" not in st.session_state:
        st.warning("Primero carga un archivo en la pestaña 📂 Datos.")
    else:
        df = st.session_state["df"]
        numericas = df.select_dtypes(include=np.number).columns.tolist()
        categoricas = df.select_dtypes(exclude=np.number).columns.tolist()

        y = st.selectbox("Variable dependiente (numérica):", numericas, key="y_fact")
        factores = st.multiselect("Selecciona los factores categóricos:", categoricas, key="fact_mult")
        interaccion = st.checkbox("Incluir interacción entre factores", value=True, key="fact_inter")

        # Construcción segura de fórmula: C(Q('col')) para variables categóricas con espacios/paréntesis
        def _citado(f):  # cita el nombre de columna para patsy
            return f"C(Q('{f}'))"

        if len(factores) >= 1:
            if interaccion and len(factores) >= 2:
                rhs = " * ".join([_citado(f) for f in factores])   # incluye interacción
            else:
                rhs = " + ".join([_citado(f) for f in factores])   # sólo efectos principales
            formula = f"{y} ~ {rhs}"
        else:
            formula = None

        if formula:
            st.markdown(f"**Fórmula generada:** `{formula}`")

        # Botón calcular
        if st.button("Calcular ANOVA factorial"):
            try:
                # 1) Asegurar tipos categóricos
                for f in factores:
                    df[f] = df[f].astype("category")

                # 2) Ajuste del modelo y ANOVA
                from statsmodels.formula.api import ols
                from statsmodels.stats.anova import anova_lm

                modelo = ols(formula, data=df).fit()
                anova_tabla = anova_lm(modelo, typ=2)
                st.success("✅ ANOVA factorial calculado correctamente.")
                st.dataframe(anova_tabla)

                # 3) Gráfico principal (Plotly) – caja por combinación si procede
                import plotly.express as px
                if len(factores) == 1:
                    f = factores[0]
                    fig = px.box(
                        df, x=f, y=y, color=f,
                        title=f"Distribución de {y} según {f}",
                        points="all",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(
                        title_font=dict(size=18, color="#002F6C", family="Arial Black"),
                        xaxis_title=f, yaxis_title=y,
                        template="simple_white", hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    # Columna de combinación para visualizar la interacción
                    comb_name = "×".join(factores)
                    df["_comb"] = df[factores].astype(str).agg(" × ".join, axis=1)
                    fig = px.box(
                        df, x="_comb", y=y, color="_comb",
                        title=f"Distribución de {y} por combinación de factores ({comb_name})",
                        points="all",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(
                        title_font=dict(size=18, color="#002F6C", family="Arial Black"),
                        xaxis_title="Combinación de niveles", yaxis_title=y,
                        template="simple_white", hovermode="x unified",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # =======================================================
                # 🔍 Pruebas post-hoc (Tukey HSD)
                # =======================================================
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                import plotly.graph_objects as go
                import numpy as np
                import pandas as pd

                st.subheader("🔍 Comparaciones múltiples (Tukey HSD)")

                # --- Tukey por factor principal (one-way por cada factor) ---
                for f in factores:
                    niveles = df[f].dropna().unique()
                    if len(niveles) < 3:
                        st.info(f"ℹ️ Tukey para **{f}** omitido (requiere ≥ 3 niveles).")
                        continue

                    st.markdown(f"**Tukey HSD — Factor:** `{f}`")
                    tuk = pairwise_tukeyhsd(endog=df[y], groups=df[f], alpha=0.05)

                    tuk_df = pd.DataFrame(
                        data=tuk._results_table.data[1:],
                        columns=tuk._results_table.data[0]
                    )
                    st.dataframe(tuk_df)

                    # Gráfico horizontal de diferencias
                    tfig = go.Figure()
                    tfig.add_trace(go.Bar(
                        x=tuk_df["meandiff"],
                        y=[f"{a} vs {b}" for a, b in zip(tuk_df["group1"], tuk_df["group2"])],
                        orientation='h',
                        marker_color=['#002F6C' if sig else '#A9A9A9' for sig in tuk_df["reject"]],
                        hovertemplate=("Comparación: %{y}<br>Diferencia media: %{x:.3f}"
                                       "<br>p-ajustada: %{customdata:.4f}"),
                        customdata=tuk_df["p-adj"]
                    ))
                    tfig.update_layout(
                        title=f"Resultados Tukey HSD — {f}",
                        title_font=dict(size=17, color="#002F6C", family="Arial Black"),
                        xaxis_title="Diferencia de medias", yaxis_title="Comparaciones",
                        template="simple_white", height=420
                    )
                    st.plotly_chart(tfig, use_container_width=True)

                    # Conclusión breve
                    n_sig = int(tuk_df["reject"].sum())
                    if n_sig > 0:
                        st.info(f"**Conclusión:** En **{f}** se detectan {n_sig} diferencias significativas (p < 0.05).")
                    else:
                        st.info(f"**Conclusión:** En **{f}** no se detectan diferencias significativas (p ≥ 0.05).")

                # --- Tukey sobre combinaciones (interacción) ---
                if len(factores) >= 2:
                    grupos_comb = df["_comb"]
                    if grupos_comb.nunique() >= 3:
                        st.markdown("**Tukey HSD — Combinaciones (interacción)**")
                        tuk_c = pairwise_tukeyhsd(endog=df[y], groups=grupos_comb, alpha=0.05)
                        tuk_c_df = pd.DataFrame(
                            data=tuk_c._results_table.data[1:],
                            columns=tuk_c._results_table.data[0]
                        )
                        st.dataframe(tuk_c_df)

                        tfigc = go.Figure()
                        tfigc.add_trace(go.Bar(
                            x=tuk_c_df["meandiff"],
                            y=[f"{a} vs {b}" for a, b in zip(tuk_c_df["group1"], tuk_c_df["group2"])],
                            orientation='h',
                            marker_color=['#003A70' if sig else '#B0B0B0' for sig in tuk_c_df["reject"]],
                            hovertemplate=("Comparación: %{y}<br>Diferencia media: %{x:.3f}"
                                           "<br>p-ajustada: %{customdata:.4f}"),
                            customdata=tuk_c_df["p-adj"]
                        ))
                        tfigc.update_layout(
                            title="Resultados Tukey HSD — Interacción (grupos combinados)",
                            title_font=dict(size=17, color="#002F6C", family="Arial Black"),
                            xaxis_title="Diferencia de medias", yaxis_title="Comparaciones",
                            template="simple_white", height=480
                        )
                        st.plotly_chart(tfigc, use_container_width=True)

                        n_sig_c = int(tuk_c_df["reject"].sum())
                        if n_sig_c > 0:
                            st.info(f"**Conclusión:** Entre combinaciones de niveles se detectan {n_sig_c} diferencias significativas (p < 0.05).")
                        else:
                            st.info("**Conclusión:** No se detectan diferencias significativas entre combinaciones de niveles (p ≥ 0.05).")
                    else:
                        st.info("ℹ️ Tukey de interacción omitido (se requieren ≥ 3 combinaciones).")

                # ===============================================================
                # 🏆 Mejor combinación de factores (interacción óptima)
                # ===============================================================
                if len(factores) >= 2 and "_comb" in df.columns and grupos_comb.nunique() >= 3:
                    st.markdown("### 🏆 Mejor combinación de factores (interacción óptima)")

                    try:
                        medias_int = df.groupby("_comb")[y].mean().sort_values(ascending=False)
                        medias_int_df = medias_int.rename("Media").round(4).reset_index()
                        top_int = medias_int_df.iloc[0, 0]

                        pmap_int, sigmap_int = {}, {}
                        if "tuk_c_df" in locals() and not tuk_c_df.empty:
                            for _, r in tuk_c_df.iterrows():
                                g1, g2 = str(r["group1"]), str(r["group2"])
                                p = float(r["p-adj"])
                                if g1 == str(top_int):
                                    pmap_int[g2] = p
                                elif g2 == str(top_int):
                                    pmap_int[g1] = p
                            for lvl in medias_int.index:
                                sigmap_int[lvl] = True if lvl == top_int else (pmap_int.get(lvl, np.nan) < 0.05)
                        else:
                            for lvl in medias_int.index:
                                sigmap_int[lvl] = (lvl == top_int)
                            st.warning("ℹ️ No se encontraron resultados Tukey válidos para la comparación entre combinaciones.")

                        import plotly.graph_objects as go
                        niveles_int = list(medias_int.index)
                        valores_int = medias_int.values
                        colores_int = ['#003A70' if sigmap_int.get(niv, False) else '#B0B0B0' for niv in niveles_int]

                        hover_text = []
                        for niv in niveles_int:
                            if niv == top_int:
                                hover_text.append(f"{niv}<br>Media: {medias_int[niv]:.4f}<br>Comparación base")
                            else:
                                ptxt = pmap_int.get(niv, np.nan)
                                if np.isnan(ptxt):
                                    hover_text.append(f"{niv}<br>Media: {medias_int[niv]:.4f}<br>p (vs {top_int}): —")
                                else:
                                    hover_text.append(f"{niv}<br>Media: {medias_int[niv]:.4f}<br>p (vs {top_int}): {ptxt:.4f}")

                        bar_int = go.Figure()
                        bar_int.add_trace(go.Bar(
                            x=niveles_int,
                            y=valores_int,
                            marker_color=colores_int,
                            hovertext=hover_text,
                            hoverinfo="text"
                        ))
                        bar_int.update_layout(
                            title=f"Medias de {y} por combinación de factores (referencia: {top_int})",
                            title_font=dict(size=18, color="#002F6C", family="Arial Black"),
                            xaxis_title="Combinación de factores",
                            yaxis_title=f"Media de {y}",
                            template="simple_white",
                            bargap=0.25
                        )

                        col1, col2 = st.columns([1.2, 2.3])
                        with col1:
                            st.dataframe(medias_int_df, use_container_width=True, height=280)
                        with col2:
                            st.plotly_chart(bar_int, use_container_width=True)

                        comparables = [niv for niv in niveles_int if niv != top_int]
                        sig_vs_top = [niv for niv in comparables if pmap_int.get(niv, np.nan) < 0.05]
                        n_comp, n_sig = len(comparables), len(sig_vs_top)

                        if n_comp == 0:
                            texto = f"""
                            <div style="background-color:#F8F9FA; border-left:4px solid #002F6C; padding:12px; border-radius:10px;">
                            <b>📘 Conclusión aplicada:</b><br>
                            Solo existe una combinación de niveles, por lo que no puede evaluarse superioridad relativa.
                            </div>
                            """
                        elif n_sig == n_comp:
                            texto = f"""
                            <div style="background-color:#F8F9FA; border-left:4px solid #002F6C; padding:12px; border-radius:10px;">
                            <b>📘 Conclusión aplicada:</b><br>
                            La combinación <b>{top_int}</b> presenta la <b>media más alta</b> de <b>{y}</b> y difiere significativamente (Tukey, p &lt; 0.05)
                            de todas las demás combinaciones. Puede considerarse la <b>interacción óptima</b> bajo las condiciones evaluadas.
                            </div>
                            """
                        elif n_sig > 0:
                            lista_diff = ", ".join(sig_vs_top)
                            texto = f"""
                            <div style="background-color:#F8F9FA; border-left:4px solid #002F6C; padding:12px; border-radius:10px;">
                            <b>📘 Conclusión aplicada:</b><br>
                            La combinación <b>{top_int}</b> tiene la <b>media más alta</b> y difiere significativamente (Tukey, p &lt; 0.05)
                            de: <b>{lista_diff}</b>. Frente a las demás combinaciones no se evidencian diferencias significativas,
                            por lo que su superioridad es <b>parcial</b>.
                            </div>
                            """
                        else:
                            texto = f"""
                            <div style="background-color:#F8F9FA; border-left:4px solid #002F6C; padding:12px; border-radius:10px;">
                            <b>📘 Conclusión aplicada:</b><br>
                            La combinación <b>{top_int}</b> posee la mayor media de <b>{y}</b>, pero no difiere significativamente
                            de las demás combinaciones (Tukey, p ≥ 0.05). No hay evidencia estadística suficiente
                            para afirmar que sea la mejor interacción.
                            </div>
                            """
                        st.markdown(texto, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"❌ Error al determinar la mejor interacción: {e}")

            except Exception as e:
                st.error(f"❌ Error al calcular ANOVA factorial: {e}")



# ===============================================================
# 📊 TAB 4 — SUPUESTOS Y CONCLUSIONES GLOBALES
# ===============================================================
with tabs[3]:
    st.header("📊 Verificación de supuestos del modelo ANOVA")

    if "model" not in st.session_state:
        st.warning("⚠️ No se ha ejecutado ningún modelo aún.")
    else:
        modelo = st.session_state["model"]
        residuales = modelo.resid

        # --- Normalidad de residuos ---
        st.subheader("📈 Normalidad de residuos (Shapiro–Wilk)")
        shapiro = stats.shapiro(residuales)
        st.write(f"Estadístico W = {shapiro.statistic:.4f}, p = {shapiro.pvalue:.4f}")
        if shapiro.pvalue > 0.05:
            st.success("✔️ Los residuos parecen normales.")
        else:
            st.warning("⚠️ Los residuos no son normales (p < 0.05).")

        # QQ-plot
        with plt.rc_context({'font.size': 8}):
            fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=120)
            sm.qqplot(residuales, line="s", ax=ax)
            ax.set_title("QQ-Plot de residuos", fontsize=8)
            st.pyplot(fig, use_container_width=False)

            # --- Homogeneidad de varianzas ---
            st.subheader("📊 Homogeneidad de varianzas (Prueba de Levene)")

            try:
                # Identificar variable dependiente (endógena)
                y_name = modelo.model.endog_names

                # Detectar factores categóricos usados en la fórmula
                factores_en_modelo = [
                    c for c in df.columns
                    if df[c].dtype == "object" or df[c].dtype.name == "category"
                ]

                # Si no hay factores categóricos, usar el primero de tipo string
                if len(factores_en_modelo) == 0:
                    factores_en_modelo = [df.columns[1]]

                # Tomar el primer factor categórico para Levene
                factor_principal = factores_en_modelo[0]

                # Agrupar por ese factor y tomar los grupos del valor de Y
                grupos = [grupo[y_name].values for _, grupo in df.groupby(factor_principal)]

                # Calcular la prueba de Levene
                lev = stats.levene(*grupos)
                st.write(f"Estadístico W = {lev.statistic:.4f}, p = {lev.pvalue:.4f}")

                if lev.pvalue > 0.05:
                    st.success("✔️ Varianzas homogéneas (se cumple el supuesto).")
                else:
                    st.warning("⚠️ Varianzas heterogéneas (posible violación del supuesto).")

            except Exception as e:
                st.error(f"❌ Error al calcular la prueba de Levene: {e}")


        # ------------------------------------------------------
        # CHECKLIST DE SUPUESTOS
        # ------------------------------------------------------
        def emoji_check(passed): return "✅" if passed else "❌"
        normal = shapiro.pvalue > 0.05
        homo = lev.pvalue > 0.05
        checklist = f"""
        <div style="background-color:#F8F9FA; border-left:4px solid {UPS_BLUE};
        padding:10px; border-radius:10px;">
        <b>Evaluación de supuestos:</b><br><br>
        <ul style="list-style:none; line-height:1.8;">
        <li>{emoji_check(normal)} <b>Normalidad (Shapiro–Wilk)</b>: {'Se cumple' if normal else 'No se cumple'}</li>
        <li>{emoji_check(homo)} <b>Homoscedasticidad (Levene)</b>: {'Se cumple' if homo else 'No se cumple'}</li>
        </ul>
        </div>
        """
        st.markdown(checklist, unsafe_allow_html=True)

        # ------------------------------------------------------
        # CONCLUSIÓN GLOBAL
        # ------------------------------------------------------
        if normal and homo:
            st.success("✔️ Todos los supuestos se cumplen. El modelo ANOVA es válido para inferencia estadística.")
        else:
            st.warning("""
            ⚠️ Algunos supuestos no se cumplen completamente.
            Considera aplicar transformaciones (logarítmica o raíz cuadrada)
            o usar pruebas no paramétricas equivalentes (Kruskal–Wallis o Friedman).
            """)
        # =======================================================
        # 🔍 ANÁLISIS POST-HOC Y ALTERNATIVAS NO PARAMÉTRICAS
        # =======================================================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("🔍 Análisis Post-Hoc y Pruebas Alternativas")

        try:
            # -------------------------------------------------------
            # 1️⃣ CASO: SUPUESTOS CUMPLIDOS → TUKEY HSD
            # -------------------------------------------------------
            if normal and homo:
                st.subheader("📊 Comparaciones múltiples — Prueba de Tukey (HSD)")
                from statsmodels.stats.multicomp import pairwise_tukeyhsd

                # Identificar variable dependiente y factor principal
                y_name = modelo.model.endog_names
                factor_categorico = [
                    c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"
                ][0]

                # Ejecutar Tukey
                tukey = pairwise_tukeyhsd(endog=df[y_name], groups=df[factor_categorico], alpha=0.05)

                # Mostrar resultados
                st.text(tukey.summary())

                # Visualización (gráfico de Tukey)
                fig_tukey, ax_tukey = plt.subplots(figsize=(5, 3), dpi=120)
                tukey.plot_simultaneous(ax=ax_tukey)
                ax_tukey.set_title("Comparaciones múltiples (Tukey HSD)", fontsize=9, weight="bold")
                st.pyplot(fig_tukey, use_container_width=False)

                st.markdown(f"""
                <div style="background-color:#F8F9FA; border-left:4px solid {UPS_BLUE};
                padding:10px; border-radius:8px;">
                <b>Interpretación:</b><br>
                • Si el intervalo de confianza no cruza cero → diferencia significativa entre medias.<br>
                • Las comparaciones con p < 0.05 indican grupos estadísticamente diferentes.<br>
                • Este análisis solo es válido cuando los supuestos del ANOVA se cumplen.<br>
                </div>
                """, unsafe_allow_html=True)

            # -------------------------------------------------------
            # 2️⃣ CASO: SUPUESTOS NO CUMPLIDOS → PRUEBAS NO PARAMÉTRICAS
            # -------------------------------------------------------
            else:
                st.subheader("⚠️ Supuestos no cumplidos — Pruebas no paramétricas recomendadas")

                # Detectar número de factores categóricos
                cat_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"]
                y_name = modelo.model.endog_names

                if len(cat_cols) == 1:
                    st.markdown("✅ Aplicando **Kruskal–Wallis** (alternativa no paramétrica al ANOVA de un factor)")
                    grupos = [grupo[y_name].values for _, grupo in df.groupby(cat_cols[0])]
                    kw = stats.kruskal(*grupos)
                    st.write(f"Estadístico H = {kw.statistic:.4f}, p = {kw.pvalue:.4f}")

                    if kw.pvalue < 0.05:
                        st.success("✔️ Se detectan diferencias significativas entre grupos (p < 0.05).")
                    else:
                        st.info("⚠️ No se detectan diferencias significativas (p ≥ 0.05).")

                elif len(cat_cols) > 1:
                    st.markdown("✅ Aplicando **Friedman** (diseños con medidas repetidas o factoriales)")
                    # Friedman requiere datos balanceados por sujeto/condición
                    try:
                        pivot_df = df.pivot_table(index=cat_cols[0], columns=cat_cols[1], values=y_name)
                        stat, p = stats.friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])
                        st.write(f"Estadístico χ² = {stat:.4f}, p = {p:.4f}")
                        if p < 0.05:
                            st.success("✔️ Diferencias significativas detectadas entre condiciones (p < 0.05).")
                        else:
                            st.info("⚠️ No se detectan diferencias significativas (p ≥ 0.05).")
                    except Exception as e:
                        st.error(f"❌ No se pudo aplicar Friedman: {e}")

                st.markdown(f"""
                <div style="background-color:#F9FAFB; border-left:4px solid #6C63FF;
                padding:10px; border-radius:8px;">
                <b>Conclusión:</b><br>
                • Cuando los supuestos de normalidad u homocedasticidad no se cumplen, se utilizan pruebas no paramétricas.<br>
                • <b>Kruskal–Wallis</b> evalúa diferencias entre grupos independientes.<br>
                • <b>Friedman</b> evalúa diferencias en diseños con medidas repetidas o factores combinados.<br>
                • Estas pruebas son más robustas frente a distribuciones no normales.<br>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error en el análisis post-hoc o pruebas no paramétricas: {e}")

