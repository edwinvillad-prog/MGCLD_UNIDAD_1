# app_bondad_ajuste_consolidada.py
# Aplicación profesional en Streamlit — Bondad de ajuste y validación avanzada
# Fases 1, 2 y 3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import (
    chisquare, poisson, binom, hypergeom,
    norm, expon, gamma, weibull_min, lognorm,
    kstest, anderson
)
from math import comb

# Para la prueba de Lilliefors (solo Normalidad) disponible en statsmodels
from statsmodels.stats.diagnostic import kstest_normal


# =========================
# CONFIG GENERAL
# =========================
st.set_page_config(
    page_title="Bondad de Ajuste y Validación — Posgrado",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Laboratorio interactivo de bondad de ajuste y validación")
st.markdown("### Maestría en Gestión de la Calidad de la Leche y sus Derivados")

# =========================
# SIDEBAR (solo una vez)
# =========================
st.sidebar.header("Opciones generales")
st.sidebar.markdown("👤 **M.Sc. Edwin Villarreal, Fís.** ")

alpha = st.sidebar.selectbox("Nivel de significancia (α)", [0.01, 0.05, 0.10], index=1)

# =========================
# CARGA DE ARCHIVO (robusta)
# =========================
st.sidebar.markdown("### 📂 Cargar archivo")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV/Excel", type=["csv", "xlsx"])
data = None

if uploaded_file:
    try:
        # --- Lectura automática según extensión ---
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # --- 🔧 Limpieza de encabezados ---
        data.columns = data.columns.map(lambda c: str(c).strip())

        # --- Mostrar confirmación y vista previa ---
        st.sidebar.success(f"✅ Archivo cargado correctamente: {uploaded_file.name}")
        st.sidebar.write("**Columnas detectadas:**")
        st.sidebar.dataframe(data.head(3))

        # --- Verificación rápida de tipo de dato ---
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.sidebar.warning("⚠️ No se detectaron columnas numéricas. "
                               "Revise el formato o los encabezados de su archivo.")

    except Exception as e:
        st.sidebar.error(f"❌ Error al leer el archivo: {e}")
else:
    st.sidebar.info("⚠️ Cargue un archivo CSV o Excel con encabezados en la primera fila.")


decision_mode_disc = st.sidebar.radio(
    "Modo de decisión (discretas)", 
    ["Docente", "Estricto/Industrial"],
    key="decision_mode_disc"
)

decision_mode_cont = st.sidebar.radio(
    "Modo de decisión (continuas)",
    ["Docente", "Estricto/Industrial"],
    index=0,
    key="decision_mode_cont"
)


# =========================
# TABS PRINCIPALES
# =========================
tabs = st.tabs([
    "📘 Teoría",
    "🎲 Distribuciones discretas",
    "📈 Distribuciones continuas",
    "🧮 Validación avanzada discreta",
    "📊 Validación avanzada continua",
    "⚠️ Escenarios críticos"
])

# =========================
# TAB 1 — TEORÍA
# =========================
with tabs[0]:
    st.header("📘 Teoría y condiciones de aplicación")

    # Distribuciones discretas
    st.subheader("🎲 Distribuciones discretas")
    st.markdown("""
    - **Chi²** → comparación de frecuencias observadas y esperadas.  
      - **Tipo:** prueba **no paramétrica**.  
      - **Condiciones:** $n ≥ 30$, frecuencias esperadas ≥ 5.  
      - **Aplicación en lácteos:** conteo de **UFC (Unidades Formadoras de Colonias)**, defectos en envases.  

      **Fórmula del estadístico:**  
      $$
      \\chi^2 = \\sum_{i=1}^k \\frac{(O_i - E_i)^2}{E_i}
      $$
      donde $O_i$ son las frecuencias observadas y $E_i$ las esperadas.  

    - **Poisson**: conteo de microorganismos.  
    - **Binomial**: rechazo de lotes con muestras defectuosas.  
    - **Hipergeométrica**: muestreo sin reemplazo.  

    ⚠️ *Si $n < 20$ o hay frecuencias esperadas < 5 → aplicar prueba exacta o simulación Monte Carlo.*
    """)

    # Distribuciones continuas
    st.subheader("📈 Distribuciones continuas")
    st.markdown("""
    - **Normal**: proteína, grasa, densidad, pH.  
    - **Exponencial**: tiempos de falla, vida útil.  
    - **Weibull**: confiabilidad, vida útil de quesos/yogures.  
    - **Gamma**: tiempos de fermentación.  
    - **Lognormal**: procesos de degradación.  

    - **Kolmogorov–Smirnov (KS)**  
      - **Tipo:** prueba **no paramétrica**.  
      - **Condiciones:** muestra continua, $n ≥ 20$.  
      - **Fórmula:**  
        $$
        D = \\sup_x \\left| F_n(x) - F(x) \\right|
        $$  
        donde $F_n(x)$ es la CDF empírica y $F(x)$ la teórica.  

    - **Lilliefors** (variante de KS)  
      - **Tipo:** no paramétrica.  
      - Se aplica cuando los parámetros de la distribución se **estiman de la muestra** (ej. Normal, Exponencial).  

    - **Shapiro–Wilk**  
      - **Tipo:** paramétrica.  
      - Diseñada para Normalidad con **muestras pequeñas (n < 20)**.  

    - **Anderson–Darling (AD)**  
      - **Tipo:** no paramétrica.  
      - Requiere $n ≥ 8$.  
      - Mayor sensibilidad en las colas de la distribución.  
      - Muy usada en **vida útil y confiabilidad**.  
    """)

    # Validación y regla práctica
    st.subheader("🧪 Validación práctica")
    st.markdown("""
    - **Chi²** → discretas.  
    - **Exacta/Fisher** → discretas con $n < 20$ o esperados < 5.  
    - **KS clásico** → continuas (n ≥ 20, parámetros conocidos).  
    - **Lilliefors** → Normal/Exponencial con parámetros estimados.  
    - **Shapiro–Wilk** → Normalidad en muestras pequeñas (n < 20).  
    - **Anderson–Darling** → n ≥ 8, sensible en colas.  

    🔍 Además de las pruebas numéricas, se recomienda inspección visual mediante **gráficos P–P y Q–Q**.

    ✅ **Regla práctica:**  
    - Si $p > α$ → **No se rechaza H₀** (ajuste adecuado).  
    - Si $p ≤ α$ → **Se rechaza H₀** (el modelo no explica bien los datos).  
    """)

# =========================
# TAB 2 — DISTRIBUCIONES DISCRETAS
# =========================
with tabs[1]:
    st.header("🎲 Ajuste de distribuciones discretas")
        # Banner con contexto actual
    st.info(f"🔎 Nivel de significancia actual: **α = {alpha}**\n\n"
            f"📌 Modo de decisión: **{decision_mode_disc}**")

    if data is not None:
       
        # --- 🔧 Mostrar solo columnas numéricas ---
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("❌ No hay columnas numéricas en el archivo. "
                    "Revise que el archivo tenga datos numéricos (sin texto).")
            st.stop()

        variable = st.selectbox("Seleccione variable discreta", numeric_cols, key="var_disc")


        dist_choice = st.radio("Distribución de referencia", ["Poisson", "Binomial", "Hipergeométrica"], key="dist_disc")

        x = data[variable].dropna().values
        # --- 🔧 Validación de discreción ---
        if np.any(x % 1 != 0):
            st.warning("⚠️ La variable seleccionada tiene valores decimales. "
                    "Se redondearán al entero más cercano para aplicar la prueba discreta.")
            x = np.round(x).astype(int)
        else:
            x = x.astype(int)

        x = np.clip(x, 0, None)  # Evita valores negativos

        obs_counts = np.bincount(x)
        n = len(x)

        # ⚠️ Advertencia automática si n < 20
        small_sample = False
        if n < 20:
            small_sample = True
            st.warning("⚠️ Tamaño de muestra muy pequeño (n < 20). "
                       "El test Chi² puede tener poca validez. "
                       "Interprete los resultados con precaución.")

        if dist_choice == "Poisson":
            lam = np.mean(x)
            exp_counts = [poisson.pmf(k, lam) * n for k in range(len(obs_counts))]
            dist_info = f"Poisson(λ={lam:.2f})"

        elif dist_choice == "Binomial":
            n_trials = st.number_input("Número de ensayos (n)", 1, 100, 10)
            p = np.mean(x) / n_trials
            exp_counts = [binom.pmf(k, n_trials, p) * n for k in range(len(obs_counts))]
            dist_info = f"Binomial(n={n_trials}, p={p:.2f})"

        else:  # Hipergeométrica
            N = st.number_input("Tamaño población (N)", 1, 1000, 50)
            K = st.number_input("Éxitos en población (K)", 1, int(N), 10)
            n_sample = st.number_input("Tamaño de muestra (n)", 1, int(N), 10)
            exp_counts = [hypergeom.pmf(k, N, K, n_sample) * n for k in range(len(obs_counts))]
            dist_info = f"Hipergeométrica(N={N}, K={K}, n={n_sample})"

        exp_counts = np.array(exp_counts)
        exp_counts *= n / exp_counts.sum()

        # ⚠️ Advertencia si hay frecuencias esperadas < 5
        low_expected = np.sum(exp_counts < 5)
        if low_expected > 0:
            st.warning(f"⚠️ Se detectaron {low_expected} categorías con frecuencias esperadas < 5. "
                       "Esto puede afectar la validez del test Chi².")

        # =========================
        # 1) Chi² clásico
        # =========================
        chi2, pval = chisquare(obs_counts, f_exp=exp_counts)
        st.subheader("📈 Chi² clásico")
        st.write(f"Estadístico = {chi2:.3f}, p-valor = {pval:.4f}")

        # =========================
        # 2) Monte Carlo (simulación p-valor)
        # =========================
        def chi2_montecarlo(obs, exp, n_sim=5000):
            chi2_obs, _ = chisquare(obs, f_exp=exp)
            probs = exp / np.sum(exp)
            n_tot = np.sum(obs)
            sims = []
            for _ in range(n_sim):
                sim_data = np.random.multinomial(n_tot, probs)
                chi2_sim, _ = chisquare(sim_data, f_exp=exp)
                sims.append(chi2_sim)
            pval_mc = np.mean([s >= chi2_obs for s in sims])
            return chi2_obs, pval_mc

        chi2_mc, pval_mc = None, None
        if low_expected > 0 or small_sample:
            chi2_mc, pval_mc = chi2_montecarlo(obs_counts, exp_counts, n_sim=5000)
            st.subheader("🎲 Chi² con Monte Carlo")
            st.write(f"Estadístico = {chi2_mc:.3f}, p-valor simulado = {pval_mc:.4f}")

        # =========================
        # 3) Opciones adicionales
        # =========================
        if small_sample or low_expected > 0:
            st.markdown("### 🔧 Opciones adicionales de validación")

            # --- Reagrupar categorías automáticamente ---
            if st.button("📊 Reagrupar categorías automáticamente"):
                obs = obs_counts.copy()
                exp = exp_counts.copy()

                while np.any(exp < 5) and len(exp) > 1:
                    if exp[0] < 5:  
                        exp[1] += exp[0]; obs[1] += obs[0]
                        exp, obs = exp[1:], obs[1:]
                    elif exp[-1] < 5:  
                        exp[-2] += exp[-1]; obs[-2] += obs[-1]
                        exp, obs = exp[:-1], obs[:-1]
                    else:
                        idx = np.argmin(exp)
                        if idx > 0:
                            exp[idx-1] += exp[idx]; obs[idx-1] += obs[idx]
                            exp, obs = np.delete(exp, idx), np.delete(obs, idx)
                        else:
                            exp[1] += exp[0]; obs[1] += obs[0]
                            exp, obs = exp[1:], obs[1:]

                chi2_adj, pval_adj = chisquare(obs, f_exp=exp)
                st.write("### 📊 Chi² tras reagrupar categorías")
                st.write(f"Estadístico = {chi2_adj:.3f}, p-valor = {pval_adj:.4f}")
                if pval_adj < alpha:
                    st.error("❌ Rechazar H₀ tras reagrupar.")
                else:
                    st.success("✅ No rechazar H₀ tras reagrupar.")

            # --- Prueba exacta ---
            if st.button("🎯 Aplicar prueba exacta"):
                from scipy.stats import binomtest, fisher_exact
                if dist_choice == "Binomial":
                    k = np.sum(x)  
                    n_total = len(x)
                    p0 = np.mean(x) / n_trials if n_trials > 0 else 0.5
                    test_res = binomtest(k, n_total, p=p0)
                    st.write(f"**Prueba Binomial exacta:** éxitos={k}, n={n_total}, p-valor = {test_res.pvalue:.4f}")
                    if test_res.pvalue < alpha:
                        st.error("❌ Rechazar H₀ (Binomial exacta).")
                    else:
                        st.success("✅ No rechazar H₀ (Binomial exacta).")
                elif len(obs_counts) == 2:  
                    tabla = np.array([[obs_counts[0], obs_counts[1]],
                                      [n - obs_counts[0], n - obs_counts[1]]])
                    odds, p_fisher = fisher_exact(tabla)
                    st.write(f"**Prueba de Fisher (2x2):** p-valor = {p_fisher:.4f}")
                    if p_fisher < alpha:
                        st.error("❌ Rechazar H₀ (Fisher).")
                    else:
                        st.success("✅ No rechazar H₀ (Fisher).")
                else:
                    st.info("ℹ️ La prueba exacta está implementada solo para Binomial y tablas 2×2.")

            # --- Aumentar tamaño de muestra ---
            if st.button("➕ Aumentar tamaño de muestra"):
                st.info("📌 Recolecte más observaciones (ideal n ≥ 30). "
                        "Con más datos, las frecuencias esperadas aumentan y el test Chi² gana potencia estadística.")

        # =========================
        # 📌 Dictamen final
        # =========================
        st.markdown("---")
        st.subheader("🧾 Dictamen final")

        chi2_reject = (pval < alpha)
        mc_reject = (pval_mc < alpha) if pval_mc is not None else None

        if decision_mode_disc == "Estricto/Industrial":
            # Si alguna prueba primaria rechaza → rechazo
            if (chi2_reject or (mc_reject is True)):
                driver = "Monte Carlo" if mc_reject else "Chi² clásico"
                st.error(f"**Dictamen final (estricto):** Rechazar H₀ (α={alpha}). "
                         f"Decisión basada en **{driver}**.")
            else:
                base = "Monte Carlo" if mc_reject is False else "Chi² clásico"
                st.success(f"**Dictamen final (estricto):** No rechazar H₀ (α={alpha}). "
                           f"Decisión basada en **{base}**.")

        else:  # Docente
            # Priorizar Chi² clásico; Monte Carlo como apoyo
            if chi2_reject:
                st.error(f"**Dictamen final (docente):** Rechazar H₀ (α={alpha}). Según Chi² clásico (p={pval:.4f}).")
            else:
                st.success(f"**Dictamen final (docente):** No rechazar H₀ (α={alpha}). Según Chi² clásico (p={pval:.4f}).")
                if mc_reject is not None and mc_reject != chi2_reject:
                    st.info("ℹ️ Nota: Monte Carlo llegó a una conclusión distinta (sensibilidad a esperados bajos).")

        # =========================
        # Gráfico
        # =========================
        fig = go.Figure()
        fig.add_bar(x=list(range(len(obs_counts))), y=obs_counts, name="Observados")
        fig.add_scatter(x=list(range(len(exp_counts))), y=exp_counts, mode="lines+markers",
                        name="Esperados", line=dict(color="red"))
        fig.update_layout(title=f"Chi² — {dist_info}", xaxis_title="Valores", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)






# =========================
# TAB 3 — DISTRIBUCIONES CONTINUAS
# =========================
with tabs[2]:
    st.header("📈 Ajuste de distribuciones continuas")
        # Banner con contexto actual
    st.info(f"🔎 Nivel de significancia actual: **α = {alpha}**\n\n"
            f"📌 Modo de decisión: **{decision_mode_cont}**")

    if data is not None:
        # --- 🔧 Mostrar solo columnas numéricas ---
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("❌ No hay columnas numéricas en el archivo. "
                    "Revise que el archivo tenga datos numéricos (sin texto).")
            st.stop()

        variable = st.selectbox("Seleccione variable continua", numeric_cols, key="var_cont")

        dist_choice = st.radio("Distribución de referencia", ["Normal", "Exponencial", "Weibull", "Gamma", "Lognormal"], key="dist_cont")

        x = data[variable].dropna().values
        n = len(x)

        # Advertencia automática si el tamaño muestral es pequeño
        if n < 20:
            st.warning("⚠️ Tamaño de muestra muy pequeño (n < 20). "
                       "Las pruebas de bondad de ajuste pueden tener poca potencia estadística. "
                       "Interprete los resultados con precaución y considere aumentar la muestra.")

        # Inicializamos banderas
        have_lillie = False
        have_shap = False
        have_ad = False
        ad_pass = None  # para usar en dictamen final

        # =========================
        # NORMAL
        # =========================
        if dist_choice == "Normal":
            mu, sigma = np.mean(x), np.std(x, ddof=1)
            D, pval = kstest(x, "norm", args=(mu, sigma))
            dist_info = f"Normal(μ={mu:.2f}, σ={sigma:.2f})"
    
            # ➕ Prueba Lilliefors (usando kstest_normal de statsmodels)
            from statsmodels.stats.diagnostic import kstest_normal
            stat_lillie, pval_lillie = kstest_normal(x)
            have_lillie = True

            # ➕ Shapiro–Wilk (solo si n < 20)
            if n < 20:
                from scipy.stats import shapiro
                stat_shap, pval_shap = shapiro(x)
                have_shap = True
                st.subheader("🧮 Shapiro–Wilk (n < 20)")
                st.write(f"Estadístico={stat_shap:.3f}, p-valor={pval_shap:.4f}")
                if pval_shap < alpha:
                    st.error(f"❌ Según Shapiro–Wilk, se rechaza H₀ al nivel α={alpha}. "
                             "Los datos no parecen provenir de una distribución Normal.")
                else:
                    st.success(f"✅ Según Shapiro–Wilk, no se rechaza H₀. "
                               "Los datos son consistentes con una distribución Normal.")

            # ➕ Anderson–Darling (n ≥ 8)
            if n >= 8:
                from scipy.stats import anderson
                ad_res = anderson(x, dist="norm")
                have_ad = True
                st.subheader("📊 Anderson–Darling (colas de la distribución)")
                st.write(f"Estadístico AD={ad_res.statistic:.3f}")
                niveles = ad_res.significance_level / 100
                idx = (np.abs(niveles - alpha)).argmin()
                valor_critico = ad_res.critical_values[idx]
                st.write(f"Valor crítico (α={alpha}): {valor_critico:.3f}")
                if ad_res.statistic > valor_critico:
                    ad_pass = False
                    st.error(f"❌ Según Anderson–Darling, se rechaza H₀ al nivel α={alpha}. "
                             "Los datos no siguen Normal, con énfasis en colas.")
                else:
                    ad_pass = True
                    st.success(f"✅ Según Anderson–Darling, no se rechaza H₀ al nivel α={alpha}. "
                               "Los datos son consistentes con una distribución Normal.")

        # =========================
        # EXPONENCIAL
        # =========================
        elif dist_choice == "Exponencial":
            lam = 1 / np.mean(x)
            D, pval = kstest(x, "expon", args=(0, 1/lam))
            dist_info = f"Exponencial(λ={lam:.2f})"

            # ➕ Prueba Lilliefors casera (Monte Carlo con parámetros estimados)
            def lilliefors_expon(x, n_sim=500):
                lam = 1 / np.mean(x)
                D, _ = kstest(x, 'expon', args=(0, 1/lam))
                n = len(x)
                sims = []
                for _ in range(n_sim):
                    sample = np.random.exponential(scale=1/lam, size=n)
                    D_sim, _ = kstest(sample, 'expon', args=(0, 1/lam))
                    sims.append(D_sim)
                pval = np.mean(np.array(sims) >= D)
                return D, pval

            stat_lillie, pval_lillie = lilliefors_expon(x)
            have_lillie = True

            # ➕ Anderson–Darling (n ≥ 8)
            if n >= 8:
                from scipy.stats import anderson
                ad_res = anderson(x, dist="expon")
                have_ad = True
                st.subheader("📊 Anderson–Darling (colas de la distribución)")
                st.write(f"Estadístico AD={ad_res.statistic:.3f}")
                niveles = ad_res.significance_level / 100
                idx = (np.abs(niveles - alpha)).argmin()
                valor_critico = ad_res.critical_values[idx]
                st.write(f"Valor crítico (α={alpha}): {valor_critico:.3f}")
                if ad_res.statistic > valor_critico:
                    ad_pass = False
                    st.error(f"❌ Según Anderson–Darling, se rechaza H₀ al nivel α={alpha}. "
                             "Los datos no siguen Exponencial, con énfasis en colas.")
                else:
                    ad_pass = True
                    st.success(f"✅ Según Anderson–Darling, no se rechaza H₀ al nivel α={alpha}. "
                               "Los datos son consistentes con una distribución Exponencial.")

        # =========================
        # OTRAS DISTRIBUCIONES
        # =========================
        elif dist_choice == "Weibull":
            c, loc, scale = weibull_min.fit(x, floc=0)
            D, pval = kstest(x, "weibull_min", args=(c, loc, scale))
            dist_info = f"Weibull(β={c:.2f}, escala={scale:.2f})"

        elif dist_choice == "Gamma":
            a, loc, scale = gamma.fit(x, floc=0)
            D, pval = kstest(x, "gamma", args=(a, loc, scale))
            dist_info = f"Gamma(k={a:.2f}, θ={scale:.2f})"

        else:  # Lognormal
            s, loc, scale = lognorm.fit(x, floc=0)
            D, pval = kstest(x, "lognorm", args=(s, loc, scale))
            dist_info = f"Lognormal(s={s:.2f}, escala={scale:.2f})"

        # =========================
        # Resultados KS
        # =========================
        st.subheader("📈 Kolmogorov–Smirnov (KS clásico)")
        st.write(f"Estadístico KS={D:.3f}")
        st.write(f"p-valor={pval:.4f}")

        if pval < alpha:
            st.error(f"❌ Según KS clásico, se rechaza H₀. Los datos **no siguen** {dist_info}.")
        else:
            st.success(f"✅ Según KS clásico, no se rechaza H₀. Los datos **siguen** {dist_info}.")

        # =========================
        # Resultados Lilliefors (solo Normal y Exponencial)
        # =========================
        if dist_choice in ["Normal", "Exponencial"] and have_lillie:
            st.subheader("🔹 Lilliefors (parámetros estimados)")
            st.write(f"Estadístico={stat_lillie:.3f}, p-valor={pval_lillie:.4f}")
            if pval_lillie < alpha:
                st.error(f"❌ Según Lilliefors, se rechaza H₀ al nivel α={alpha}.")
            else:
                st.success(f"✅ Según Lilliefors, no se rechaza H₀. El ajuste a {dist_choice} es consistente.")

        # =========================
        # 📌 DICTAMEN FINAL (criterio seleccionable)
        # =========================
        ks_reject = (pval < alpha)

        lillie_reject = None
        if dist_choice in ["Normal", "Exponencial"] and have_lillie:
            lillie_reject = (pval_lillie < alpha)

        shap_reject = None
        if dist_choice == "Normal" and have_shap:
            shap_reject = (pval_shap < alpha)

        ad_reject = None
        if have_ad:
            ad_reject = (ad_pass is False)  # AD no da p-valor; usamos crítico

        primary_results = []

        if dist_choice == "Normal":
            if n < 20 and have_shap:
                primary_results.append(("Shapiro–Wilk", shap_reject, f"p={pval_shap:.4f}"))
            else:
                if have_ad:
                    primary_results.append(("Anderson–Darling", ad_reject, "comparación con valor crítico"))
                if have_lillie:
                    primary_results.append(("Lilliefors", lillie_reject, f"p={pval_lillie:.4f}"))
                primary_results.append(("KS clásico", ks_reject, f"p={pval:.4f}"))

        elif dist_choice == "Exponencial":
            if have_ad:
                primary_results.append(("Anderson–Darling", ad_reject, "comparación con valor crítico"))
            if have_lillie:
                primary_results.append(("Lilliefors", lillie_reject, f"p={pval_lillie:.4f}"))
            primary_results.append(("KS clásico", ks_reject, f"p={pval:.4f}"))

        else:
            primary_results.append(("KS clásico", ks_reject, f"p={pval:.4f}"))

        st.markdown("---")
        st.subheader("🧾 Dictamen final")

        if decision_mode_cont.startswith("Estricto"):
            reject_any = any(r for _, r, _ in primary_results if r is True)
            driver = next((name for name, r, _ in primary_results if r is True), primary_results[0][0])

            if reject_any:
                st.error(f"**Dictamen final (estricto):** Rechazar H₀ (α={alpha}). "
                         f"Decisión basada en **{driver}**.")
            else:
                base_name, _, base_detail = primary_results[0]
                st.success(f"**Dictamen final (estricto):** No rechazar H₀ (α={alpha}). "
                           f"Decisión basada en **{base_name}** ({base_detail}).")

        else:
            if any(name == "Lilliefors" for name, _, _ in primary_results):
                name, rej, detail = next((n, r, d) for n, r, d in primary_results if n == "Lilliefors")
            else:
                name, rej, detail = primary_results[0]

            if rej is True:
                st.error(f"**Dictamen final (docente):** Rechazar H₀ (α={alpha}). "
                         f"Decisión basada en **{name}** ({detail}).")
            else:
                st.success(f"**Dictamen final (docente):** No rechazar H₀ (α={alpha}). "
                           f"Decisión basada en **{name}** ({detail}).")

        # 4) Nota de discrepancias
        flags = []
        if have_ad and ad_reject is not None and ad_reject != ks_reject:
            flags.append("AD y KS llegan a conclusiones diferentes (colas vs centro).")
        if lillie_reject is not None and lillie_reject != ks_reject:
            flags.append("Lilliefors y KS difieren (parámetros estimados).")
        if shap_reject is not None and dist_choice == "Normal" and n < 20 and shap_reject != ks_reject:
            flags.append("Shapiro–Wilk y KS difieren (n pequeño).")

        if flags:
            st.warning("**Discrepancias detectadas:** " + " ".join(flags))

        # Tabla resumen de p-valores
        summary_rows = [{"Prueba": "KS clásico", "p-valor": pval}]
        if have_lillie:
            summary_rows.append({"Prueba": "Lilliefors", "p-valor": pval_lillie})
        if have_shap:
            summary_rows.append({"Prueba": "Shapiro–Wilk", "p-valor": pval_shap})
        if have_ad:
            summary_rows.append({"Prueba": "Anderson–Darling", "p-valor": "Basado en crítico"})

        st.markdown("**Resumen de p-valores (referencia):**")
        st.table(pd.DataFrame(summary_rows))

        # =========================
        # Gráfico
        # =========================
        hist = np.histogram(x, bins=20, density=True)
        fig = go.Figure()
        fig.add_bar(x=hist[1][:-1], y=hist[0], name="Datos", opacity=0.6)
        xs = np.linspace(min(x), max(x), 200)
        if dist_choice == "Normal":
            ys = norm.pdf(xs, mu, sigma)
        elif dist_choice == "Exponencial":
            ys = expon.pdf(xs, 0, 1/lam)
        elif dist_choice == "Weibull":
            ys = weibull_min.pdf(xs, c, loc, scale)
        elif dist_choice == "Gamma":
            ys = gamma.pdf(xs, a, loc, scale)
        else:
            ys = lognorm.pdf(xs, s, loc, scale)
        fig.add_scatter(x=xs, y=ys, mode="lines", name=f"{dist_choice} teórica", line=dict(color="red"))
        fig.update_layout(title=f"Ajuste {dist_choice}", xaxis_title="Valores", yaxis_title="Densidad")
        st.plotly_chart(fig, use_container_width=True)


# =========================
# TAB NUEVO — VALIDACIÓN DISCRETA AVANZADA
# =========================
with tabs[3]:
    st.header("🧮 Validación avanzada de modelos discretos")

    # Banner de contexto (α y modo de decisión para discretas)
    st.info(f"🔎 Nivel de significancia actual: **α = {alpha}**\n\n"
            f"📌 Modo de decisión: **{decision_mode_disc}**")

    st.markdown("Esta sección valida **modelos discretos** (Poisson, Binomial, Hipergeométrica) "
                "usando log-verosimilitud, AIC/BIC y análisis de residuos (Pearson y deviance). "
                "Se complementa con Chi² y opción Monte Carlo cuando aplique.")

    if data is not None:
        # Selección de variable y distribución
        # --- 🔧 Mostrar solo columnas numéricas ---
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("❌ No hay columnas numéricas en el archivo. "
                    "Verifique que las columnas contengan datos numéricos (conteos o frecuencias).")
            st.stop()

        variable = st.selectbox("Seleccione variable discreta", numeric_cols, key="val_disc_var")

        dist_choice = st.radio("Modelo discreto a validar",
                               ["Poisson", "Binomial", "Hipergeométrica"],
                               key="val_disc_dist")

        # Vector de observaciones discretas
        x = data[variable].dropna().values
        # --- 🔧 Validación de discreción ---
        if np.any(x % 1 != 0):
            st.warning("⚠️ La variable seleccionada tiene valores decimales. "
                    "Se redondearán al entero más cercano para aplicar el modelo discreto.")
            x = np.round(x).astype(int)
        else:
            x = x.astype(int)

        x = np.clip(x, 0, None)

        n = len(x)

        # Conteos observados por categoría (0,1,2,...,k)
        obs_counts = np.bincount(x)
        k_vals = np.arange(len(obs_counts))

        # Advertencia por tamaño muestral
        if n < 20:
            st.warning("⚠️ Tamaño muestral pequeño (n < 20). "
                       "Chi² puede ser inestable; use Monte Carlo o pruebas exactas cuando aplique.")

        # -------------------------------------------------
        # 1) Parámetros, loglike y probabilidades teóricas
        # -------------------------------------------------
        if dist_choice == "Poisson":
            lam = np.mean(x)
            pmf = poisson.pmf(k_vals, lam)
            loglike = np.sum(poisson.logpmf(x, lam))
            param_text = f"λ̂ = {lam:.4f}"
            k_params = 1

        elif dist_choice == "Binomial":
            # El usuario debe especificar n_trials (ensayos por observación)
            n_trials = st.number_input("Número de ensayos por observación (n)", 1, 1000, 10, key="val_disc_ntrials")
            # Estimación p̂ = media(x)/n_trials (supuesto clásico)
            p = np.clip(np.mean(x) / n_trials, 1e-9, 1-1e-9)
            pmf = binom.pmf(k_vals, n_trials, p)
            loglike = np.sum(binom.logpmf(x, n_trials, p))
            param_text = f"n = {int(n_trials)}, p̂ = {p:.4f}"
            k_params = 1  # (asumiendo n_trials fijo y p estimado)

        else:  # Hipergeométrica
            # Parametrización: N población, K éxitos en población, n_s tamaño de muestra por observación
            N = st.number_input("Tamaño población (N)", 1, 100000, 200, key="val_disc_N")
            K = st.number_input("Éxitos en población (K)", 0, int(N), min(int(N)//4, int(N)), key="val_disc_K")
            n_s = st.number_input("Tamaño de muestra por observación (n_s)", 1, int(N), min(10, int(N)), key="val_disc_ns")
            # pmf en soporte observado
            pmf = hypergeom.pmf(k_vals, N, K, n_s)
            # log-like: producto de pmf de cada observación individual
            # (si x contiene conteos de éxitos en submuestras de tamaño n_s)
            with np.errstate(divide='ignore'):
                loglike = np.sum(hypergeom.logpmf(x, N, K, n_s))
            param_text = f"N={int(N)}, K={int(K)}, n_s={int(n_s)}"
            k_params = 2  # (K y n_s como dados por usuario → si los tratas como estimados, ajustar)

        # Normaliza por seguridad (evitar pequeños errores numéricos)
        pmf = np.clip(pmf, 1e-15, 1.0)
        pmf = pmf / pmf.sum()

        # -------------------------------------------------
        # 2) AIC/BIC
        # -------------------------------------------------
        aic = -2 * loglike + 2 * k_params
        bic = -2 * loglike + k_params * np.log(n)

        st.subheader("📋 Log-verosimilitud y criterios de información")
        st.table(pd.DataFrame({"Estadístico": ["Loglike", "AIC", "BIC", "Parámetros"],
                               "Valor": [loglike, aic, bic, param_text]}))

        # -------------------------------------------------
        # 3) Esperados, Chi² y Monte Carlo (si aplica)
        # -------------------------------------------------
        exp_counts = pmf * n
        # Ajuste de normalización por seguridad
        exp_counts *= n / exp_counts.sum()

        # Chequeo de esperados bajos
        low_expected = int(np.sum(exp_counts < 5))
        if low_expected > 0:
            st.warning(f"⚠️ Se detectaron {low_expected} categorías con esperados < 5. "
                       "El Chi² puede perder validez. Considere **reagrupar** o usar **Monte Carlo**.")

        chi2, pval = chisquare(obs_counts, f_exp=exp_counts)

        st.subheader("📈 Chi² clásico (referencia)")
        st.write(f"Estadístico = {chi2:.3f}, p-valor = {pval:.4f}")

        # Monte Carlo para p-valor de Chi² cuando n es pequeño o hay esperados bajos
        pval_mc = None
        if (n < 20) or (low_expected > 0):
            st.caption("🎲 Monte Carlo activado por n pequeño o esperados bajos.")
            def chi2_montecarlo(obs, exp, n_sim=5000):
                chi2_obs, _ = chisquare(obs, f_exp=exp)
                probs = exp / np.sum(exp)
                n_tot = int(np.sum(obs))
                sims = np.empty(n_sim)
                for i in range(n_sim):
                    sim_data = np.random.multinomial(n_tot, probs)
                    sims[i], _ = chisquare(sim_data, f_exp=exp)
                return np.mean(sims >= chi2_obs)

            pval_mc = chi2_montecarlo(obs_counts, exp_counts, n_sim=5000)
            st.write(f"**Chi² Monte Carlo:** p-valor simulado = {pval_mc:.4f}")

        # -------------------------------------------------
        # 4) Residuos (Pearson y Deviance) y gráficos
        # -------------------------------------------------
        st.subheader("📊 Residuos (diagnóstico)")

        # Pearson residuals
        with np.errstate(divide='ignore', invalid='ignore'):
            pearson_res = (obs_counts - exp_counts) / np.sqrt(np.where(exp_counts > 0, exp_counts, np.nan))

        # Deviance residuals aproximados para conteos (evitar 0*log(0))
        dev_terms = np.where(obs_counts > 0,
                             2 * obs_counts * np.log(np.where(exp_counts > 0, obs_counts / exp_counts, np.nan)),
                             0.0)
        deviance = np.nansum(dev_terms)

        st.write(f"Deviance total (aprox.): {deviance:.3f}")

        # Gráfico: Observado vs Esperado
        fig_oe = go.Figure()
        fig_oe.add_bar(x=k_vals, y=obs_counts, name="Observados")
        fig_oe.add_scatter(x=k_vals, y=exp_counts, mode="lines+markers", name="Esperados", line=dict(color="red"))
        fig_oe.update_layout(title=f"Observado vs Esperado — {dist_choice}",
                             xaxis_title="Clases", yaxis_title="Frecuencia")
        st.plotly_chart(fig_oe, use_container_width=True)

        # Gráfico: Residuos de Pearson
        fig_pr = go.Figure()
        fig_pr.add_bar(x=k_vals, y=np.nan_to_num(pearson_res), name="Residuos de Pearson")
        fig_pr.update_layout(title="Residuos de Pearson por clase",
                             xaxis_title="Clases", yaxis_title="(O-E)/√E")
        st.plotly_chart(fig_pr, use_container_width=True)

        # -------------------------------------------------
        # 5) Dictamen final (alineado con Tab 2)
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("🧾 Dictamen final")

        chi2_reject = (pval < alpha)
        mc_reject = (pval_mc is not None) and (pval_mc < alpha)

        if decision_mode_disc.startswith("Estricto"):
            # Estricto/Industrial: si alguna evidencia principal rechaza → rechazar
            if chi2_reject or mc_reject:
                driver = "Monte Carlo" if mc_reject else "Chi² clásico"
                st.error(f"**Dictamen final (estricto):** Rechazar H₀ (α={alpha}). "
                         f"Decisión basada en **{driver}**.")
            else:
                base = "Monte Carlo" if pval_mc is not None else "Chi² clásico"
                st.success(f"**Dictamen final (estricto):** No rechazar H₀ (α={alpha}). "
                           f"Decisión basada en **{base}**.")

        else:
            # Docente: prioriza Chi²; Monte Carlo es apoyo
            if chi2_reject:
                st.error(f"**Dictamen final (docente):** Rechazar H₀ (α={alpha}). "
                         f"Según Chi² (p={pval:.4f}).")
            else:
                st.success(f"**Dictamen final (docente):** No rechazar H₀ (α={alpha}). "
                           f"Según Chi² (p={pval:.4f}).")
                if pval_mc is not None and (mc_reject != chi2_reject):
                    st.info("ℹ️ Nota: Monte Carlo llegó a una conclusión distinta; sensible a esperados bajos.")

        # Nota de uso de AIC/BIC (comparativo)
        st.caption("📌 **AIC/BIC** son comparativos entre **modelos**: valores más bajos indican mejor ajuste relativo. "
                   "Use esta sección para comparar Poisson vs Binomial vs Hipergeométrica ante la misma variable.")




# =========================
# TAB 5 — VALIDACIÓN AVANZADA
# =========================
with tabs[4]:
    st.header("📊 Validación avanzada de modelos continuos")

    # 1. Banner con contexto (α y modo de decisión)
    st.info(f"🔎 Nivel de significancia actual: **α = {alpha}**\n\n"
            f"📌 Modo de decisión: **{decision_mode_cont}**")

    st.markdown("ℹ️ Esta validación avanzada aplica únicamente a **distribuciones continuas** "
                "(Normal, Exponencial, Weibull). Se analizan log-verosimilitud, criterios de información "
                "y gráficos Q–Q, P–P.")

    if data is not None:
        variable = st.selectbox("Seleccione variable", data.columns, key="val_av")
        dist_choice = st.radio("Distribución", ["Normal", "Exponencial", "Weibull"], key="val_dist")
        x = data[variable].dropna().values
        n = len(x)

        # 4. Advertencia para muestras pequeñas
        if n < 20:
            st.warning("⚠️ Tamaño muestral pequeño (n < 20). "
                       "Los criterios AIC/BIC y las gráficas Q–Q, P–P pueden ser poco confiables. "
                       "Considere aumentar el tamaño de muestra.")

        # Ajuste de parámetros y log-verosimilitud
        if dist_choice == "Normal":
            mu, sigma = np.mean(x), np.std(x, ddof=1)
            loglike = np.sum(norm.logpdf(x, mu, sigma))
            params = (mu, sigma)
            theo_q = norm.ppf(np.linspace(0.01, 0.99, n), *params)

        elif dist_choice == "Exponencial":
            lam = 1/np.mean(x)
            loglike = np.sum(expon.logpdf(x, 0, 1/lam))
            params = (0, 1/lam)
            theo_q = expon.ppf(np.linspace(0.01, 0.99, n), *params)

        else:  # Weibull
            c, loc, scale = weibull_min.fit(x, floc=0)
            loglike = np.sum(weibull_min.logpdf(x, c, loc, scale))
            params = (c, loc, scale)
            theo_q = weibull_min.ppf(np.linspace(0.01, 0.99, n), *params)

        # Criterios de información
        aic = -2*loglike + 2*len(params)
        bic = -2*loglike + len(params)*np.log(n)

        # 2. Resultados con subtítulo más claro
        st.subheader("📋 Resultados de log-verosimilitud y criterios de información")
        st.table(pd.DataFrame({"Estadístico": ["Loglike", "AIC", "BIC"],
                               "Valor": [loglike, aic, bic]}))

        # Q-Q Plot
        st.subheader("📉 Gráfico Q–Q")
        obs_q = np.sort(x)
        fig = go.Figure()
        fig.add_scatter(x=theo_q, y=obs_q, mode="markers", name="Datos")
        fig.add_scatter(x=theo_q, y=theo_q, mode="lines", name="45°", line=dict(dash="dash"))
        fig.update_layout(title=f"Gráfico Q–Q ({dist_choice})", xaxis_title="Cuantiles teóricos", yaxis_title="Cuantiles observados")
        st.plotly_chart(fig, use_container_width=True)

        # P-P Plot
        st.subheader("📈 Gráfico P–P")
        ecdf = np.arange(1, n+1)/n
        if dist_choice == "Normal":
            cdf_theo = norm.cdf(np.sort(x), mu, sigma)
        elif dist_choice == "Exponencial":
            cdf_theo = expon.cdf(np.sort(x), 0, 1/lam)
        else:
            cdf_theo = weibull_min.cdf(np.sort(x), c, loc, scale)

        fig2 = go.Figure()
        fig2.add_scatter(x=cdf_theo, y=ecdf, mode="markers", name="Datos")
        fig2.add_scatter(x=[0,1], y=[0,1], mode="lines", name="45°", line=dict(dash="dash"))
        fig2.update_layout(title=f"Gráfico P–P ({dist_choice})", xaxis_title="CDF teórica", yaxis_title="CDF empírica")
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Dictamen final en base a AIC/BIC
        st.markdown("---")
        st.subheader("🧾 Dictamen final")

        if aic < bic:
            st.success(f"✅ El modelo {dist_choice} parece razonable: AIC={aic:.2f}, BIC={bic:.2f}. "
                       "Un AIC más bajo sugiere mejor ajuste relativo.")
        else:
            st.warning(f"⚠️ El modelo {dist_choice} no es óptimo: AIC={aic:.2f} ≥ BIC={bic:.2f}. "
                       "Considere probar otras distribuciones o validar con datos adicionales.")

        # 5. Interpretaciones aplicadas
        if dist_choice == "Normal":
            st.info("📌 Una buena alineación en los gráficos Q–Q y P–P confirma que las fluctuaciones "
                    "siguen un patrón gaussiano. Esto es esperado en parámetros como proteína o densidad de la leche.")
        elif dist_choice == "Exponencial":
            st.info("📌 Un buen ajuste exponencial indica que los tiempos hasta eventos (fallas, vida útil) "
                    "se comportan como procesos de memoria nula, típico en microbiología o fallas aleatorias.")
        else:
            st.info("📌 En Weibull, el parámetro β permite interpretar el tipo de falla: "
                    "β<1 (tempranas), β≈1 (azarosas), β>1 (desgaste).")

# =========================
# TAB 5 — ESCENARIOS CRÍTICOS
# =========================
with tabs[5]:
    st.header("⚠️ Escenarios críticos en industria láctea")
    st.caption("🏭 Enfoque industrial: cada decisión estadística se interpreta como riesgo o medida de control en procesos lácteos.")
    st.info("🔍 En esta sección se simulan **problemas reales de la industria láctea**. "
            "Se aplican distribuciones para modelar riesgos y se interpretan los resultados con base en estadísticos, "
            "p-valores y curvas de supervivencia.")

    if data is not None:
        escenario = st.selectbox("Seleccione escenario", 
                                 ["Contaminación microbiológica", "Falla en cadena de frío", "Comparación vida útil"])
        variable = st.selectbox("Variable a analizar", data.columns, key="var_scen")
        x = data[variable].dropna().values
        n = len(x)

        # =========================
        # ESCENARIO 1: CONTAMINACIÓN MICROBIOLÓGICA
        # =========================
        if escenario == "Contaminación microbiológica":
            lam = np.mean(x)

            # --- 🔧 Corrección: validar si los datos son enteros ---
            if np.any(x % 1 != 0):
                st.warning("⚠️ La variable seleccionada tiene valores decimales. "
                        "Se redondearán al entero más cercano para aplicar el modelo de Poisson.")
                x = np.round(x).astype(int)
            else:
                x = x.astype(int)

            # Evitar valores negativos (Poisson no los permite)
            x = np.clip(x, 0, None)

            # --- Cálculo corregido de esperados ---
            exp_counts = [poisson.pmf(k, lam) * len(x) for k in range(int(max(x)) + 1)]
            exp_counts = np.array(exp_counts)
            exp_counts *= len(x) / exp_counts.sum()
            obs_counts = np.bincount(x, minlength=len(exp_counts))


            chi2, pval = chisquare(obs_counts, f_exp=exp_counts)

            st.subheader("📈 Chi² clásico (Poisson)")
            st.write(f"Estadístico = {chi2:.3f}, p-valor = {pval:.4f}")

            # Monte Carlo si n es pequeño o hay esperados bajos
            low_expected = np.sum(exp_counts < 5)
            pval_mc = None
            if n < 20 or low_expected > 0:
                st.caption("🎲 Monte Carlo activado automáticamente (n pequeño o esperados bajos).")

                def chi2_montecarlo(obs, exp, n_sim=5000):
                    chi2_obs, _ = chisquare(obs, f_exp=exp)
                    probs = exp / np.sum(exp)
                    n_tot = np.sum(obs)
                    sims = []
                    for _ in range(n_sim):
                        sim_data = np.random.multinomial(n_tot, probs)
                        chi2_sim, _ = chisquare(sim_data, f_exp=exp)
                        sims.append(chi2_sim)
                    pval_mc = np.mean([s >= chi2_obs for s in sims])
                    return chi2_obs, pval_mc

                chi2_mc, pval_mc = chi2_montecarlo(obs_counts, exp_counts)
                st.subheader("🎲 Chi² con Monte Carlo")
                st.write(f"Estadístico = {chi2_mc:.3f}, p-valor simulado = {pval_mc:.4f}")

            # Dictamen final
            st.subheader("🧾 Dictamen final")
            if (pval < alpha) or (pval_mc is not None and pval_mc < alpha):
                st.error(f"❌ Rechazar H₀ (α={alpha}). Posible riesgo microbiológico detectado.")
            else:
                st.success(f"✅ No rechazar H₀ (α={alpha}). Conteos consistentes con Poisson (evento raro).")

            st.caption("📌 Interpretación: Si el modelo Poisson es válido, los conteos de UFC corresponden "
                       "a eventos raros independientes. Un rechazo de H₀ sugiere contaminación sistemática o fallas higiénicas.")

            # Gráfico Observados vs Esperados
            fig = go.Figure()
            fig.add_bar(x=list(range(len(obs_counts))), y=obs_counts, name="Observados")
            fig.add_scatter(x=list(range(len(exp_counts))), y=exp_counts, mode="lines+markers",
                            name="Esperados", line=dict(color="red"))
            fig.update_layout(title="Chi² — Ajuste Poisson", xaxis_title="Conteos", yaxis_title="Frecuencia")
            st.plotly_chart(fig, use_container_width=True)

        # =========================
        # ESCENARIO 2: FALLA EN CADENA DE FRÍO
        # =========================
        elif escenario == "Falla en cadena de frío":
            c, loc, scale = weibull_min.fit(x, floc=0)
            loglike = np.sum(weibull_min.logpdf(x, c, loc, scale))

            st.subheader("📊 Ajuste Weibull")
            st.write(f"Weibull(β={c:.2f}, escala={scale:.2f}) — Loglike={loglike:.2f}")
            st.info("🔹 Interpretación: β<1 fallas tempranas, β≈1 azarosas, β>1 desgaste.")
            st.caption("📌 Contexto técnico: Identificar si la pérdida de frío genera fallas prematuras (β<1) "
                       "o deterioros por envejecimiento acelerado (β>1).")

            # Gráfico supervivencia Weibull
            xs = np.linspace(min(x), max(x), 200)
            fig = go.Figure()
            fig.add_scatter(x=xs, y=weibull_min.sf(xs, c, loc, scale), 
                            mode="lines", name="Weibull — supervivencia", line=dict(color="blue"))
            fig.update_layout(title="Curva de supervivencia Weibull",
                              xaxis_title="Tiempo", yaxis_title="S(t)")
            st.plotly_chart(fig, use_container_width=True)

            # Dictamen final
            st.subheader("🧾 Dictamen final")
            if c < 1:
                st.warning("⚠️ Riesgo de fallas tempranas: β<1 → productos sensibles a ruptura de frío.")
            elif c > 1:
                st.error("❌ Riesgo por desgaste acelerado: β>1 → deterioro progresivo en cadena de frío.")
            else:
                st.success("✅ Fallas aleatorias: β≈1 → comportamiento azaroso.")

        # =========================
        # ESCENARIO 3: COMPARACIÓN VIDA ÚTIL
        # =========================
        else:
            lam = 1/np.mean(x)
            c, loc, scale = weibull_min.fit(x, floc=0)

            xs = np.linspace(min(x), max(x), 200)
            fig = go.Figure()
            fig.add_scatter(x=xs, y=expon.sf(xs, 0, 1/lam), name="Exponencial — refrigeración")
            fig.add_scatter(x=xs, y=weibull_min.sf(xs, c, loc, scale), name="Weibull — ruptura frío")
            fig.update_layout(title="Curvas de supervivencia — Vida útil",
                              xaxis_title="Tiempo", yaxis_title="Supervivencia (S(t))")
            st.plotly_chart(fig, use_container_width=True)

            # Dictamen final
            st.subheader("🧾 Dictamen final")
            st.info("🔹 La curva más baja indica mayor riesgo de pérdida de calidad. "
                    "Comparar ambas curvas ayuda a decidir políticas de control: "
                    "reforzar refrigeración, ajustar tiempos de distribución o "
                    "establecer límites de seguridad en logística.")
                    


