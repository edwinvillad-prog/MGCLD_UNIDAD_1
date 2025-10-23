# ===========================================================
# app_lacteos_u3.py
# Laboratorio interactivo — Unidad 3
# Correlación, regresión múltiple y validación con ANOVA
# M.Sc. Edwin Villarreal, Fís. — UPS
# ===========================================================
# AUTO-INSTALADOR DE DEPENDENCIAS FALTANTES
# ===========================================================

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necesario para proyección 3D)
# === Imports añadidos para pruebas de supuestos ===
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

# -------------------------------------------------------
# CONFIGURACIÓN INICIAL
# -------------------------------------------------------
st.set_page_config(page_title="Laboratorio — Unidad 3", layout="wide")
st.title("📊 Laboratorio Interactivo Modelamiento Estadístico")
st.sidebar.markdown("---", unsafe_allow_html=True)

# Autor justo debajo del título
st.markdown(
    "<h4 style='color:black;'>👨‍🏫 Autor: M.Sc. Edwin Villarreal, Fís.</h4>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---", unsafe_allow_html=True)

st.markdown("""
Este laboratorio permite **explorar correlaciones**, construir un **modelo de regresión lineal múltiple**
y validarlo con **ANOVA** en el contexto de **control de calidad de leche cruda**.
""")

# -------------------------------------------------------
# CARGA DE DATOS
# -------------------------------------------------------
st.sidebar.header("⚙️ Configuración")
archivo = st.sidebar.file_uploader("Sube un archivo de datos", type=["xlsx", "xls", "csv"])

if archivo is not None:
    # Carga de archivo
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo).dropna()
    else:
        df = pd.read_excel(archivo).dropna()

    # Selección de variables
    st.sidebar.subheader("🔎 Selección de variables")
    y_var = st.sidebar.selectbox("Variable dependiente (Y)", df.columns)
    X_vars = st.sidebar.multiselect(
        "Variables independientes (X)",
        [c for c in df.columns if c != y_var]
    )

    if len(X_vars) > 0:

        # -------------------------------------------------------
        # PERSISTENCIA ENTRE PESTAÑAS (mantiene los modelos)
        # -------------------------------------------------------
        if "model" not in st.session_state:
            st.session_state.model = None
        if "model_final" not in st.session_state:
            st.session_state.model_final = None
        if "sig_vars" not in st.session_state:
            st.session_state.sig_vars = []
        if "df_model" not in st.session_state:
            st.session_state.df_model = pd.DataFrame()

        # -------------------------------------------------------
        # PESTAÑAS
        # -------------------------------------------------------
        tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📂 Datos",
            "📊 Correlación",
            "📈 Modelo",
            "🧮 ANOVA & Resumen",
            "📉 Gráfico de regresión",
            "✅ Validación de supuestos"
        ])

        # -------------------------------------------------------
        # TAB 0 — DATOS
        # -------------------------------------------------------
        with tab0:
            st.subheader("📂 Vista previa de los datos")
            st.dataframe(df.head())

            st.markdown("""
            **Conclusión:** La correcta selección y limpieza de datos es la **base del análisis estadístico**.
            Se verifican valores completos y variables relevantes para evitar sesgos en los resultados.
            """)

        # -------------------------------------------------------
        # TAB 1 — CORRELACIÓN (Estilo Profesional Seaborn)
        # -------------------------------------------------------
        with tab1:
            st.markdown(
                "<h3 style='text-align:center; color:#1E1E1E;'>🔗 Matriz de correlaciones (Pearson)</h3>",
                unsafe_allow_html=True
            )

            import io
            try:
                # Solo variables numéricas
                df_corr = df[[y_var] + X_vars].select_dtypes(include=["number"]).dropna()
                if df_corr.shape[1] < 2:
                    st.warning("⚠️ No hay suficientes variables numéricas para calcular correlaciones de Pearson.")
                else:
                    # MATRIZ DE CORRELACIÓN (HEATMAP PRINCIPAL)
                    corr = df_corr.corr(method="pearson")
                    fig, ax = plt.subplots(figsize=(4.8, 3.6))
                    sns.heatmap(
                        corr, annot=True, cmap="rocket", vmin=0, vmax=1, linewidths=0.3,
                        fmt=".2f", annot_kws={"size": 9, "weight": "bold"},
                        cbar_kws={"shrink": 0.8, "label": "Coeficiente de correlación"}, ax=ax
                    )
                    ax.set_title(
                        "Matriz de correlación — Variables seleccionadas",
                        fontsize=11, weight="bold", color="#1E1E1E", pad=10
                    )
                    ax.tick_params(labelsize=9, labelcolor="#333333")
                    plt.tight_layout()

                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        st.pyplot(fig, use_container_width=False)

                    # MATRIZ DE DISPERSIÓN
                    st.markdown(
                        "<h4 style='text-align:center; color:#1E1E1E;'>📈 Matriz de dispersión y correlaciones visuales</h4>",
                        unsafe_allow_html=True
                    )

                    num_vars = len(df_corr.columns)
                    base_size = 1.8
                    fig_size = max(4.5, min(base_size * num_vars, 8.0))

                    g = sns.PairGrid(df_corr, diag_sharey=False, height=base_size, aspect=1)
                    g.map_lower(sns.scatterplot, color="#1f77b4", s=15, alpha=0.6)
                    g.map_diag(sns.histplot, kde=True, color="#1f77b4", alpha=0.7)

                    def corr_text(x, y, **kwargs):
                        r = np.corrcoef(x, y)[0, 1]
                        ax = plt.gca()
                        ax.set_axis_off()
                        ax.text(
                            0.5, 0.5, f"{r:.2f}",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=10, fontweight="bold", color="black",
                            bbox=dict(
                                facecolor=sns.color_palette("RdBu_r", as_cmap=True)(0.5 + 0.5 * r),
                                edgecolor="none", boxstyle="circle,pad=0.35"
                            )
                        )

                    g.map_upper(corr_text)
                    g.fig.set_size_inches(fig_size, fig_size)
                    plt.subplots_adjust(wspace=0.05, hspace=0.05)
                    g.fig.suptitle(
                        "Matriz de dispersión y correlaciones de Pearson",
                        y=1.02, fontsize=13, weight="bold"
                    )

                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        st.pyplot(g.fig, use_container_width=False)

                    buf = io.BytesIO()
                    g.fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                    st.download_button(
                        label="📥 Descargar matriz de dispersión (PNG)",
                        data=buf.getvalue(),
                        file_name="matriz_dispersion_correlaciones.png",
                        mime="image/png",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error al calcular la correlación: {e}")

            st.markdown("""
            <div style="background-color:#F8F9FA; border-left:4px solid #002F6C;
            padding:10px; border-radius:8px;">
            <b>Conclusión:</b><br>
            • La matriz de correlación muestra asociaciones <b>lineales</b> entre las variables seleccionadas.<br>
            • Es un paso previo para decidir qué predictores incluir en el modelo estadístico.<br>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background-color:#F9FAFB; border-left:4px solid #6C63FF;
            padding:10px; border-radius:8px;">
            <b>Interpretación:</b><br>
            • Los valores cercanos a <b>1</b> indican correlaciones lineales positivas fuertes.<br>
            • Los valores cercanos a <b>-1</b> indican correlaciones negativas fuertes.<br>
            • Los valores cercanos a <b>0</b> muestran ausencia de relación lineal.<br>
            • La matriz de dispersión inferior revela relaciones no lineales o posibles atípicos.<br>
            </div>
            """, unsafe_allow_html=True)
        # -------------------------------------------------------
        # TAB 2 — MODELO DE REGRESIÓN MÚLTIPLE
        # -------------------------------------------------------
        with tab2:
            st.subheader("📈 Modelo de regresión múltiple")

            # --- Validar existencia de variable dependiente ---
            if y_var not in df.columns:
                st.error(f"La variable dependiente '{y_var}' no se encuentra en el DataFrame.")
                st.stop()

            # --- Validar tipo de variable dependiente ---
            if not np.issubdtype(df[y_var].dtype, np.number):
                st.error("❌ La variable dependiente seleccionada no es numérica.")
                st.stop()

            # --- Filtrar solo variables numéricas ---
            X_numeric = [x for x in X_vars if np.issubdtype(df[x].dtype, np.number)]
            if len(X_numeric) == 0:
                st.warning("⚠️ No se seleccionaron variables numéricas válidas para el modelo.")
                st.stop()

            # --- Construcción de fórmula ---
            formula = f"{y_var} ~ " + " + ".join(X_numeric)
            st.markdown(f"**Fórmula generada:** {formula}")

            # --- Ajustar el modelo ---
            try:
                model = smf.ols(formula=formula, data=df).fit()
                st.session_state.model = model
                st.session_state.model_final = model
                st.session_state.df_model = df
                st.session_state.X_vars = X_numeric
                st.session_state.y_var = y_var

                st.success("✅ Modelo ajustado correctamente.")
                st.caption(f"📦 Modelo almacenado en sesión: {type(model).__name__}")

                # --- Ecuación en formato LaTeX ---
                eq_parts = [f"{y_var} = {model.params.get('Intercept', 0):.3f}"]
                for term in X_numeric:
                    coef = model.params[term]
                    sign = "+" if coef >= 0 else ""
                    eq_parts.append(f"{sign}{coef:.3f}·{term}")
                eq_str = " ".join(eq_parts)
                st.latex(eq_str)

                # --- Resumen del modelo ---
                st.write("**Resumen del modelo:**")
                st.text(model.summary())

                st.markdown("""
                **Conclusión:**  
                - El modelo permite cuantificar la relación entre las variables independientes y la variable dependiente.  
                - Los coeficientes representan el cambio esperado en la variable dependiente por unidad de cambio en el predictor, manteniendo los demás constantes.  
                """)

            except Exception as e:
                st.error(f"❌ Error al ajustar el modelo:\n\n{e}")
                st.info("Sugerencia: revisa que todas las variables seleccionadas sean numéricas.")

        # -------------------------------------------------------
        # TAB 3 — ANOVA DEL MODELO
        # -------------------------------------------------------
        with tab3:
            st.subheader("🧮 ANOVA del modelo")

            # --- Recuperar modelo ---
            model = st.session_state.get("model", None)

            if model is None:
                st.info("ℹ️ No se ha ajustado un modelo aún. Primero genera el modelo en el Tab 📈 Modelo.")
            else:
                try:
                    # --- ANOVA clásico ---
                    anova_tabla = anova_lm(model)
                    st.dataframe(anova_tabla)

                    # --- Gráfico de suma de cuadrados ---
                    with mpl.rc_context({'font.size': 8}):
                        fig2, ax2 = plt.subplots(figsize=(3.4, 2.2), dpi=120)
                        anova_tabla["sum_sq"][:-1].plot(
                            kind="bar", ax=ax2, color="skyblue",
                            edgecolor="black", linewidth=0.8
                        )
                        ax2.set_ylabel("Suma de cuadrados (SS)", fontsize=8)
                        ax2.set_title("Varianza explicada por predictor", fontsize=9, weight="bold", pad=6)
                        ax2.tick_params(axis="x", rotation=0, labelsize=7)
                        ax2.tick_params(axis="y", labelsize=7)
                        plt.tight_layout(pad=0.2)
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.pyplot(fig2, use_container_width=False)

                    # --- Resumen del modelo inicial ---
                    st.subheader("📌 Resumen del modelo inicial")
                    eq_parts = [f"{y_var} = {model.params.get('Intercept', 0):.3f}"]
                    for term in model.params.index:
                        if term != "Intercept":
                            coef = model.params[term]
                            sign = "+" if coef >= 0 else ""
                            eq_parts.append(f"{sign}{coef:.3f}·{term}")
                    eq_str_inicial = " ".join(eq_parts)
                    st.latex(eq_str_inicial)

                    sig_vars = [n for n, p in model.pvalues.items()
                                if (n != "Intercept") and (p < 0.05)]

                    st.markdown(f"""
                    - **R² ajustado:** {model.rsquared_adj:.3f}  
                    - **Predictores significativos (p < 0.05):** {sig_vars if sig_vars else "Ninguno"}
                    """)

                    # --- Modelo simplificado (si hay variables significativas) ---
                    if len(sig_vars) > 0:
                        st.subheader("📌 Resumen del modelo final (simplificado)")
                        formula_final = f"{y_var} ~ {' + '.join(sig_vars)}"
                        model_final = smf.ols(formula=formula_final, data=df).fit()
                        st.session_state.model_final = model_final
                        st.session_state.sig_vars = sig_vars

                        eq_parts_final = [f"{y_var} = {model_final.params.get('Intercept', 0):.3f}"]
                        for term in sig_vars:
                            coef = model_final.params[term]
                            sign = "+" if coef >= 0 else ""
                            eq_parts_final.append(f"{sign}{coef:.3f}·{term}")
                        eq_str_final = " ".join(eq_parts_final)
                        st.latex(eq_str_final)

                        st.markdown(f"""
                        - **R² ajustado:** {model_final.rsquared_adj:.3f}  
                        - **Predictores significativos (p < 0.05):** {sig_vars}  
                        - **Conclusión:** El modelo simplificado conserva solo las variables con influencia estadísticamente significativa.  
                        """)
                    else:
                        st.info("⚠️ No se encontraron predictores significativos para simplificar el modelo.")

                except Exception as e:
                    st.error(f"❌ Error al calcular el ANOVA: {e}")
        # -------------------------------------------------------
        # TAB 4 — GRÁFICO DE REGRESIÓN
        # -------------------------------------------------------
        with tab4:
            st.subheader("📉 Gráfico de regresión (modelo final)")

            model_final = st.session_state.get("model_final", None)
            sig_vars = st.session_state.get("sig_vars", [])
            df_model = st.session_state.get("df_model", df)

            if model_final is None:
                st.info("⚠️ No hay modelo final disponible para graficar.")
            else:
                if len(sig_vars) == 1:
                    # --- Gráfico 2D ---
                    with mpl.rc_context({'font.size': 8}):
                        fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
                        x_col = sig_vars[0]
                        y_col = y_var

                        sns.scatterplot(
                            x=df_model[x_col], y=df_model[y_col],
                            ax=ax, color="blue", s=18, alpha=0.85, label="Datos observados"
                        )

                        # --- Recta ajustada ---
                        intercept = model_final.params.get("Intercept", 0)
                        slope = model_final.params[x_col]
                        y_pred = intercept + slope * df_model[x_col]

                        ax.plot(df_model[x_col], y_pred, color="red",
                                label="Recta ajustada", linewidth=1.5)

                        # === INTERVALO DE CONFIANZA (95%) ===
                        from statsmodels.sandbox.regression.predstd import wls_prediction_std
                        prstd, iv_l, iv_u = wls_prediction_std(model_final, alpha=0.05)
                        ax.fill_between(df_model[x_col], iv_l, iv_u, color="red", alpha=0.15,
                                        label="IC 95%")

                        ax.legend(fontsize=7)
                        ax.set_title("Recta de regresión ajustada", fontsize=9, weight="bold")
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        plt.tight_layout(pad=0.2)

                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.pyplot(fig, use_container_width=False)

                        # === NUEVO BLOQUE: ecuación en formato LaTeX y resumen ===
                        st.subheader("📌 Ecuación del modelo final")

                        eq_parts_final = [f"{y_var} = {intercept:.3f}"]
                        sign = "+" if slope >= 0 else ""
                        eq_parts_final.append(f"{sign}{slope:.3f}·{x_col}")
                        eq_str_final = " ".join(eq_parts_final)
                        st.latex(eq_str_final)

                        st.markdown(f"""
                        - **R² ajustado:** {model_final.rsquared_adj:.3f}  
                        - **Predictor significativo (p < 0.05):** `{x_col}`  
                        - **Conclusión:** El modelo lineal representa adecuadamente la tendencia observada, con intervalo de confianza al 95 %.  
                        """)

                elif len(sig_vars) == 2:
                    # --- Gráfico 3D ---
                    with mpl.rc_context({'font.size': 8}):
                        fig3d = plt.figure(figsize=(3.8, 2.4), dpi=120)
                        ax = fig3d.add_subplot(111, projection='3d')
                        X1, X2 = df_model[sig_vars[0]], df_model[sig_vars[1]]
                        Y = df_model[y_var]

                        ax.scatter(X1, X2, Y, color="blue", s=12, alpha=0.85, label="Datos observados")
                        x1_grid, x2_grid = np.meshgrid(
                            np.linspace(X1.min(), X1.max(), 20),
                            np.linspace(X2.min(), X2.max(), 20)
                        )
                        y_grid = (model_final.params.get("Intercept", 0)
                                + model_final.params[sig_vars[0]] * x1_grid
                                + model_final.params[sig_vars[1]] * x2_grid)
                        ax.plot_surface(x1_grid, x2_grid, y_grid, color="red", alpha=0.4)
                        ax.set_xlabel(sig_vars[0])
                        ax.set_ylabel(sig_vars[1])
                        ax.set_zlabel(y_var)
                        ax.set_title("Plano de regresión ajustado", fontsize=9, weight="bold")

                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.pyplot(fig3d, use_container_width=False)

                        # --- Ecuación multivariable en LaTeX ---
                        st.subheader("📌 Ecuación del modelo final (formato LaTeX)")

                        eq_parts_final = [f"{y_var} = {model_final.params.get('Intercept', 0):.3f}"]
                        for term in sig_vars:
                            coef = model_final.params[term]
                            sign = "+" if coef >= 0 else ""
                            eq_parts_final.append(f"{sign}{coef:.3f}·{term}")
                        eq_str_final = " ".join(eq_parts_final)
                        st.latex(eq_str_final)

                        st.markdown(f"""
                        - **R² ajustado:** {model_final.rsquared_adj:.3f}  
                        - **Predictores significativos (p < 0.05):** {sig_vars}  
                        - **Conclusión:** El modelo multivariable conserva las variables con influencia estadísticamente significativa.  
                        """)

                else:
                    # --- Bondad de ajuste ---
                    with mpl.rc_context({'font.size': 8}):
                        fig6, ax6 = plt.subplots(figsize=(3.4, 2.2), dpi=120)
                        sns.scatterplot(
                            x=model_final.fittedvalues, y=df_model[y_var],
                            ax=ax6, color="blue", s=18, alpha=0.85, label="Observados"
                        )
                        ax6.plot(
                            [df_model[y_var].min(), df_model[y_var].max()],
                            [df_model[y_var].min(), df_model[y_var].max()],
                            color="red", linestyle="--", linewidth=0.9, label="Recta ideal"
                        )
                        ax6.set_title("Bondad de ajuste del modelo", fontsize=9, weight="bold")
                        ax6.legend(fontsize=7)
                        plt.tight_layout(pad=0.2)

                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.pyplot(fig6, use_container_width=False)

                # --- Interpretación visual ---
                st.markdown("""
                <div style="background-color:#F8F9FA; border-left:4px solid #002F6C;
                padding:10px; border-radius:8px;">
                <b>Interpretación:</b><br>
                • La pendiente positiva indica una relación directa entre la variable dependiente y el predictor principal.<br>
                • Cuanto más cercanos los puntos a la línea o al plano, mejor ajuste del modelo.<br>
                • El intervalo de confianza (banda roja) muestra la incertidumbre del ajuste al 95 %.<br>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="background-color:#F9FAFB; border-left:4px solid #6C63FF;
                padding:10px; border-radius:8px;">
                <b>Conclusión:</b><br>
                • El modelo lineal captura adecuadamente la tendencia general de los datos.<br>
                • Desviaciones grandes pueden sugerir efectos no lineales o interacción entre variables.<br>
                • Este gráfico permite validar visualmente la coherencia del modelo y la precisión del ajuste.<br>
                </div>
                """, unsafe_allow_html=True)



        # -------------------------------------------------------
        # TAB 5 — VALIDACIÓN DE SUPUESTOS
        # -------------------------------------------------------
        with tab5:
            st.subheader("✅ Validación de supuestos del modelo final")

            model_final = st.session_state.get("model_final", None)
            if model_final is None:
                st.info("⚠️ No hay modelo final disponible para validar supuestos.")
            else:
                resid = model_final.resid
                fitted = model_final.fittedvalues

                # --- Linealidad y Homocedasticidad ---
                st.subheader("📉 Linealidad y Homocedasticidad — Residuos vs Ajustados")
                with mpl.rc_context({'font.size': 8}):
                    fig7, ax7 = plt.subplots(figsize=(3.4, 2.2), dpi=120)
                    sns.scatterplot(
                        x=fitted, y=resid, ax=ax7,
                        color="purple", s=18, alpha=0.85, edgecolor="white"
                    )
                    ax7.axhline(0, color="red", linestyle="--", linewidth=0.9)
                    ax7.set_xlabel("Valores ajustados", fontsize=8)
                    ax7.set_ylabel("Residuos", fontsize=8)
                    plt.tight_layout(pad=0.2)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.pyplot(fig7, use_container_width=False)
                st.markdown("✔️ Si los residuos se dispersan aleatoriamente alrededor de 0, se cumple la linealidad y homocedasticidad.")

                # --- Breusch–Pagan ---
                st.subheader("📊 Homocedasticidad — Breusch–Pagan")
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, model_final.model.exog)
                st.markdown(f"**p = {bp_pvalue:.3f}**")
                if bp_pvalue > 0.05:
                    st.success("✔️ No se rechaza H₀ → Varianza constante (homocedasticidad aceptada).")
                else:
                    st.warning("⚠️ Se rechaza H₀ → Posible heterocedasticidad.")

                # --- Linealidad (RESET) ---
                st.subheader("📈 Linealidad — RESET Test")
                reset = linear_reset(model_final, power=2, use_f=True)
                st.markdown(f"**F = {reset.fvalue:.3f}, p = {reset.pvalue:.3f}**")
                if reset.pvalue > 0.05:
                    st.success("✔️ No se rechaza H₀ → Especificación lineal adecuada.")
                else:
                    st.warning("⚠️ Se rechaza H₀ → Posible no linealidad o términos omitidos.")

                # --- Normalidad ---
                st.subheader("📐 Normalidad de los residuos — Shapiro–Wilk y QQ-Plot")
                with mpl.rc_context({'font.size': 8}):
                    fig8, ax8 = plt.subplots(1, 2, figsize=(4.0, 2.0), dpi=120)
                    sns.histplot(resid, kde=True, ax=ax8[0], color="skyblue")
                    ax8[0].set_title("Histograma de residuos", fontsize=8)
                    qqplot(resid, line="s", ax=ax8[1])
                    ax8[1].set_title("QQ-Plot", fontsize=8)
                    plt.tight_layout(pad=0.2)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.pyplot(fig8, use_container_width=False)

                sh_stat, sh_p = shapiro(resid)
                st.markdown(f"**Shapiro–Wilk p = {sh_p:.3f}**")
                if sh_p > 0.05:
                    st.success("✔️ No se rechaza H₀ → Residuos normales.")
                else:
                    st.warning("⚠️ Se rechaza H₀ → Residuos no normales.")

                # --- Independencia (Durbin–Watson) ---
                st.subheader("🔄 Independencia de residuos — Durbin–Watson")
                dw = durbin_watson(resid)
                st.markdown(f"**Durbin–Watson = {dw:.3f}** (≈2 sugiere independencia)")
                if 1.5 < dw < 2.5:
                    st.success("✔️ No hay evidencia de autocorrelación (independencia aceptada).")
                else:
                    st.warning("⚠️ Posible autocorrelación en los residuos.")

                # --- Checklist visual ---
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader("📋 Checklist de validación")

                def emoji_check(passed): return "✅" if passed else "❌"
                linear_pass = reset.pvalue > 0.05
                homo_pass = bp_pvalue > 0.05
                norm_pass = sh_p > 0.05
                indep_pass = 1.5 < dw < 2.5

                checklist_md = f"""
                <div style="background-color:#F8F9FA; border-left:4px solid #002F6C;
                padding:12px; border-radius:10px;">
                <b>Evaluación de supuestos estadísticos:</b><br><br>
                <ul style="list-style:none; line-height:1.8;">
                <li>{emoji_check(linear_pass)} <b>Linealidad (RESET)</b>: {'Se cumple' if linear_pass else 'No se cumple'} — p = {reset.pvalue:.3f}</li>
                <li>{emoji_check(homo_pass)} <b>Homoscedasticidad (Breusch–Pagan)</b>: {'Se cumple' if homo_pass else 'No se cumple'} — p = {bp_pvalue:.3f}</li>
                <li>{emoji_check(norm_pass)} <b>Normalidad (Shapiro–Wilk + QQ-Plot)</b>: {'Se cumple' if norm_pass else 'No se cumple'} — p = {sh_p:.3f}</li>
                <li>{emoji_check(indep_pass)} <b>Independencia (Durbin–Watson)</b>: {'Se cumple' if indep_pass else 'No se cumple'} — DW = {dw:.3f}</li>
                </ul>
                </div>
                """
                st.markdown(checklist_md, unsafe_allow_html=True)

                # --- Conclusión global ---
                if all([linear_pass, homo_pass, norm_pass, indep_pass]):
                    st.success("✔️ Todos los supuestos se cumplen. El modelo es estadísticamente válido para inferencia y predicción.")
                else:
                    st.warning("""
                    ⚠️ Algunos supuestos no se cumplen completamente.
                    Revisa los gráficos de residuos y considera:
                    - Transformaciones logarítmicas o cuadráticas.
                    - Eliminación de atípicos o variables irrelevantes.
                    - Reajuste del modelo incluyendo interacciones o términos polinomiales.
                    """)
