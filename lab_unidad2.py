# app_bondad_ajuste.py
# Aplicación profesional en Streamlit para pruebas de bondad de ajuste — Posgrado

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, poisson, binom, hypergeom
from scipy.stats import kstest, norm, expon, lognorm
from scipy.stats import anderson

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Pruebas de Bondad de Ajuste — Posgrado",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Pruebas de Bondad de Ajuste — Posgrado")
st.markdown("### Maestría en Gestión de la Calidad de la Leche y sus Derivados")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Opciones de análisis")
alpha = st.sidebar.selectbox("Nivel de significancia (α)", [0.01, 0.05, 0.10], index=1)

# =========================
# SUBIDA DE ARCHIVO
# =========================
st.sidebar.markdown("### 📂 Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV o Excel", type=["csv", "xlsx"])

data = None
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    st.sidebar.success(f"Archivo cargado: {uploaded_file.name}")
else:
    st.sidebar.warning("⚠️ Por favor, cargue un archivo para habilitar los análisis.")

# =========================
# PESTAÑAS PRINCIPALES
# =========================
tabs = st.tabs([
    "📘 Teoría",
    "🧮 Chi-cuadrado (discretas)",
    "📈 Kolmogorov–Smirnov (continuas)",
    "📊 Anderson–Darling (continuas)"
])

# =========================
# TAB 1 — TEORÍA
# =========================
with tabs[0]:
    st.header("📘 Teoría de Pruebas de Bondad de Ajuste")

    st.markdown("""
    ## 🔹 1. Chi-cuadrado de bondad de ajuste
    - **Tipo de prueba:** no paramétrica.  
    - **Condiciones:**  
      - Datos **discretos** (conteos o frecuencias).  
      - **Tamaño muestral grande** (regla práctica: $n > 30$).  
      - **Frecuencia esperada mínima:** cada clase debe tener $E_i \\geq 5$. Si no, agrupar categorías.  
      - Se asume **independencia entre observaciones**.  
    - **Ventaja:** muy flexible, permite comparar con cualquier distribución.  
    - **Limitación:** depende del agrupamiento de clases, menos potente con muestras pequeñas.  
    - **Aplicación en lácteos:** conteo de **unidades formadoras de colonias (UFC)** en placas, defectos en envases.  

    **Fórmula del estadístico:**
    $$
    \\chi^2 = \\sum_{i=1}^k \\frac{(O_i - E_i)^2}{E_i}, 
    $$

    ---

    ## 🔹 2. Kolmogorov–Smirnov (KS)
    - **Tipo de prueba:** no paramétrica.  
    - **Condiciones:**  
      - Datos **continuos**.  
      - Adecuada para **muestras pequeñas o medianas** (p. ej. $n > 20$, aunque puede usarse con menos).  
      - Requiere que la **distribución teórica sea completamente especificada** (parámetros conocidos). Si no, se usa la variante **Lilliefors**.  
      - Sensible a diferencias en la **parte central** de la distribución, menos en las colas.  
    - **Ventaja:** no requiere agrupación de datos.  
    - **Limitación:** si los parámetros se estiman de la muestra, el test pierde validez directa.  
    - **Aplicación en lácteos:** verificar si la distribución de la **grasa de la leche** sigue Normal, o si los niveles de proteína siguen una distribución esperada.  

    **Fórmula del estadístico:**
    $$
    D = \\sup_x \\left| F_n(x) - F(x) \\right|,
    $$
    donde $F_n(x)$ es la función de distribución empírica y $F(x)$ la teórica.  

    ---

    ## 🔹 3. Anderson–Darling (AD)
    - **Tipo de prueba:** no paramétrica.  
    - **Condiciones:**  
      - Datos **continuos**.  
      - Recomendado para muestras **pequeñas y medianas** ($n \\geq 8$).  
      - Como en KS, los parámetros de la distribución teórica deberían estar especificados (aunque existen versiones ajustadas).  
      - Da mayor peso a las **colas** de la distribución, lo que lo hace más potente en detectar desviaciones extremas.  
    - **Ventaja:** más sensible que KS cuando hay diferencias en colas.  
    - **Limitación:** los valores críticos dependen de la distribución específica (no universales como en KS).  
    - **Aplicación en lácteos:** pruebas de **vida útil (tiempo hasta deterioro)** en productos fermentados (yogur, quesos frescos).  

    **Fórmula del estadístico:**
    $$
    A^2 = -n - \\frac{1}{n} \\sum_{i=1}^n (2i-1) \\Big[ \\ln F(x_{(i)}) + \\ln(1-F(x_{(n+1-i)})) \\Big].
    $$

    ---

    ✅ **Resumen comparativo:**  
    - **Chi²** → discreta, $n$ grande, requiere $E_i \\geq 5$.  
    - **KS** → continua, parámetros conocidos, sensible en el centro, válido para muestras pequeñas.  
    - **AD** → continua, más potente que KS en colas, recomendado en vida útil y tiempos de falla.  
    """)
# =========================
# TAB 2 — CHI-CUADRADO
# =========================
with tabs[1]:
    st.header("🧮 Prueba Chi-cuadrado — Ajuste de datos discretos")

    if data is not None:
        variable = st.selectbox("Seleccione variable discreta", data.columns, key="chi_var")
        dist = st.radio("Distribución de referencia", ["Poisson", "Binomial", "Hipergeométrica"], key="chi_dist")

        valores = data[variable].value_counts().sort_index()
        n = valores.sum()

        if dist == "Poisson":
            lam = np.mean(data[variable])
            esperados = [poisson.pmf(k, lam) * n for k in range(len(valores))]
            dist_info = f"Poisson(λ={lam:.2f})"

        elif dist == "Binomial":
            kmax = valores.index.max()
            p = np.mean(data[variable]) / kmax
            esperados = [binom.pmf(k, kmax, p) * n for k in range(len(valores))]
            dist_info = f"Binomial(n={kmax}, p={p:.2f})"

        else:  # Hipergeométrica (ejemplo simple con parámetros de muestra)
            N = 100
            K = int(np.mean(data[variable]) * 2)
            n_s = int(np.median(data[variable]))
            esperados = [hypergeom.pmf(k, N, K, n_s) * n for k in range(len(valores))]
            dist_info = f"Hipergeométrica(N={N}, K={K}, n={n_s})"

        # Ajuste de normalización
        esperados = np.array(esperados) * (n / np.sum(esperados))

        chi2, pval = chisquare(valores, f_exp=esperados)

        st.write(f"**Distribución de referencia:** {dist_info}")
        st.write(f"**Estadístico Chi²:** {chi2:.3f}")
        st.write(f"**p-valor:** {pval:.3f}")

        if pval < alpha:
             st.error(
                 f"❌ Se rechaza H₀ al nivel α={alpha}. "
                f"Esto indica que los datos observados **no se ajustan** a la distribución {dist_info}, "
                " lo que sugiere posibles factores adicionales."
                )
        else:
                 st.success(
                 f"✅ No se rechaza H₀ al nivel α={alpha}. "
                 f"Los datos pueden considerarse consistentes con la distribución {dist_info}, "
                 "por lo que el modelo es adecuado en este contexto."
                )

        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        ax.bar(valores.index, valores, alpha=0.6, label="Observados")
        ax.plot(valores.index, esperados, "ro--", label="Esperados")
        ax.set_title("Chi²: Observados vs Esperados")
        ax.set_xlabel("Clases")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        st.pyplot(fig)

# =========================
# TAB 3 — KS
# =========================
with tabs[2]:
    st.header("📈 Prueba Kolmogorov–Smirnov — Ajuste de datos continuos")

    if data is not None:
        variable = st.selectbox("Seleccione variable continua (KS)", data.columns, key="ks_var")
        dist = st.radio("Distribución de referencia", ["Normal", "Exponencial", "Lognormal"], key="ks_dist")

        x = data[variable].dropna().values

        if dist == "Normal":
            mu, sigma = np.mean(x), np.std(x, ddof=1)
            D, pval = kstest(x, "norm", args=(mu, sigma))
            dist_info = f"Normal(μ={mu:.2f}, σ={sigma:.2f})"
            cdf_theoretical = norm.cdf(np.sort(x), mu, sigma)

        elif dist == "Exponencial":
            lam = 1 / np.mean(x)
            D, pval = kstest(x, "expon", args=(0, 1/lam))
            dist_info = f"Exponencial(λ={lam:.2f})"
            cdf_theoretical = expon.cdf(np.sort(x), scale=1/lam)

        else:  # Lognormal
            shape = np.std(np.log(x), ddof=1)
            scale = np.exp(np.mean(np.log(x)))
            D, pval = kstest(x, "lognorm", args=(shape, 0, scale))
            dist_info = f"Lognormal(shape={shape:.2f}, scale={scale:.2f})"
            cdf_theoretical = lognorm.cdf(np.sort(x), shape, 0, scale)

        st.write(f"**Distribución de referencia:** {dist_info}")
        st.write(f"**Estadístico D:** {D:.3f}")
        st.write(f"**p-valor:** {pval:.3f}")

        

        if pval < alpha:
             st.error(f"❌ Se rechaza H₀ al nivel α={alpha}. "
            f"Esto significa que los datos **no siguen** la distribución {dist_info} " 
            "por lo que el modelo no es adecuado en este contexto.")
        else:
                st.success(
            f"✅ No se rechaza H₀ al nivel α={alpha}. "
            f"Los datos se ajustan con la distribución {dist_info} "
        "por lo que el modelo es adecuado en este contexto."
        )

        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        ecdf = np.arange(1, len(x)+1) / len(x)
        ax.step(np.sort(x), ecdf, where="post", label="CDF empírica")
        ax.plot(np.sort(x), cdf_theoretical, "r--", label=f"CDF {dist}")
        ax.set_title("Prueba KS")
        ax.set_xlabel("x")
        ax.set_ylabel("Probabilidad acumulada")
        ax.legend()
        st.pyplot(fig)

# =========================
# TAB 4 — AD
# =========================
with tabs[3]:
    st.header("📊 Prueba Anderson–Darling — Ajuste de datos continuos")

    if data is not None:
        variable = st.selectbox("Seleccione variable continua (AD)", data.columns, key="ad_var")
        dist = st.radio("Distribución de referencia", ["Normal", "Exponencial", "Lognormal"], key="ad_dist")

        x = data[variable].dropna().values
        resultado = anderson(x, dist="norm" if dist=="Normal" else "expon")

        # Elegir valor crítico correspondiente a α
        niveles = resultado.significance_level / 100
        idx = (np.abs(niveles - alpha)).argmin()
        critico = resultado.critical_values[idx]

        st.write(f"**Distribución de referencia:** {dist}")

        st.write(f"**Estadístico A²:** {resultado.statistic:.3f}")

        # Buscar el valor crítico más cercano al α elegido
        niveles = resultado.significance_level / 100
        idx = (np.abs(niveles - alpha)).argmin()
        valor_critico = resultado.critical_values[idx]

        st.write(f"**Valor crítico al {alpha*100:.0f}%:** {valor_critico:.3f}")

        if resultado.statistic > valor_critico:
            st.error(
        f"❌ Se rechaza H₀ al nivel α={alpha}. "
        f"Los datos muestran desviaciones significativas respecto a la distribución {dist}, "
        "especialmente en las colas."
          )
        else:
             st.success(
        f"✅ No se rechaza H₀ al nivel α={alpha}. "
        f"Los datos no presentan diferencias significativas respecto a la distribución {dist}, "
        "lo que valida el modelo asumido."
         )

        # Gráfico
        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        ax.hist(x, bins=15, density=True, alpha=0.6, edgecolor="black", label="Datos")
        xx = np.linspace(min(x), max(x), 200)
        if dist == "Normal":
            ax.plot(xx, norm.pdf(xx, np.mean(x), np.std(x, ddof=1)), "r--", lw=2, label="Normal")
        elif dist == "Exponencial":
            lam = 1 / np.mean(x)
            ax.plot(xx, expon.pdf(xx, scale=1/lam), "r--", lw=2, label="Exponencial")
        else:
            shape = np.std(np.log(x), ddof=1)
            scale = np.exp(np.mean(np.log(x)))
            ax.plot(xx, lognorm.pdf(xx, shape, 0, scale), "r--", lw=2, label="Lognormal")
        ax.set_title("Prueba Anderson–Darling")
        ax.set_xlabel("x")
        ax.set_ylabel("Densidad")
        ax.legend()
        st.pyplot(fig)
