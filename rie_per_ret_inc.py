import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import exp, sqrt

# ------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------
st.set_page_config(page_title="Modelos Weibull y Riesgo en Leche", page_icon="🧮", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#002F6C;'>🧮 Aplicaciones Weibull y Riesgo en Leche Cruda</h1>
<h4 style='text-align:center; color:#F7B500;'>Universidad Politécnica Salesiana — Posgrados UPS</h4>
<hr style='border:2px solid #002F6C;'>
""", unsafe_allow_html=True)

# =========================================================
# PESTAÑAS PRINCIPALES
# =========================================================
tabs = st.tabs(["📊 Modelo Weibull", "⚙️ Ejemplo CIP", "🥛 Muestras Leche", "📈 Intervalo Wilson"])

# =========================================================
# 1️⃣ MODELO DE FALLOS WEIBULL
# =========================================================
with tabs[0]:
    st.subheader("Modelo de fallos Weibull — Confiabilidad de equipos lácteos")

    col_panel, col_plot = st.columns([1.2, 1.8])

    with col_panel:
        st.markdown("### Parámetros e interpretación en la industria láctea")
        st.markdown("**β (forma):** Indica el patrón de fallas. En bombas CIP o válvulas, β>1 refleja desgaste, β≈1 fallas aleatorias y β<1 defectos de fabricación.")
        st.markdown("**η (escala):** Vida característica del equipo, tiempo promedio donde falla el 63% de las unidades.")
        st.markdown("**t (tiempo):** Periodo de operación analizado en días.")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            beta1 = st.slider("β₁ Aleatorias", 0.5, 3.0, 1.0)
            beta2 = st.slider("β₂ Desgaste", 0.5, 3.0, 1.5)
            beta3 = st.slider("β₃ Tempranas", 0.5, 3.0, 0.7)
        with col2:
            eta1 = st.slider("η₁ (Aleatorias, días)", 5, 50, 20)
            eta2 = st.slider("η₂ (Desgaste, días)", 5, 50, 25)
            eta3 = st.slider("η₃ (Tempranas, días)", 5, 50, 12)
        t_max = st.slider("Tiempo máximo (días)", 10, 100, 40)

    with col_plot:
        def weibull_R(t, beta, eta):
            return np.exp(-(t/eta)**beta)

        t = np.linspace(0, t_max, 200)
        fig, ax = plt.subplots(figsize=(6,4))
        for beta, eta, label in zip([beta1, beta2, beta3], [eta1, eta2, eta3], ["Aleatorias", "Desgaste", "Tempranas"]):
            ax.plot(t, weibull_R(t, beta, eta), lw=2, label=f"β={beta:.2f}, η={eta:.1f} ({label})")
        ax.set_xlabel("t (días)")
        ax.set_ylabel("R(t)")
        ax.set_title("Curvas de Confiabilidad Weibull")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.latex(r"R(t) = e^{-(t/\eta)^{\beta}}")
        st.markdown("**Interpretación:** La función R(t) indica la probabilidad de que un equipo lácteo continúe operativo después del tiempo t. Una confiabilidad baja sugiere riesgo de interrupciones en procesos CIP o de pasteurización.")

        # Tabla resumen
        st.markdown("### Resumen de parámetros ingresados")
        st.table({
            'Tipo de falla': ['Aleatorias', 'Desgaste', 'Tempranas'],
            'β (forma)': [beta1, beta2, beta3],
            'η (escala)': [eta1, eta2, eta3]
        })

# =========================================================
# 2️⃣ EJEMPLO: BOMBA CIP
# =========================================================
with tabs[1]:
    st.subheader("Ejemplo: Bomba CIP — Evaluación del riesgo de fallos")

    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.markdown("### Parámetros de entrada e interpretación")
        st.markdown("**β:** Determina si las fallas son aleatorias o por desgaste (β>1 → deterioro mecánico).")
        st.markdown("**η:** Vida media esperada de la bomba antes del fallo.")
        st.markdown("**t:** Tiempo operativo o de revisión (en días).")
        st.markdown("**Costo:** Pérdida económica por parada no planificada (USD).")

        beta = st.number_input("β (forma)", 0.1, 5.0, 1.5)
        eta = st.number_input("η (escala, días)", 1.0, 100.0, 25.0)
        costo = st.number_input("Costo por parada (USD)", 0.0, 10000.0, 1200.0)
        t_fail = st.slider("Tiempo de evaluación (días)", 1, 100, 20)

    with col2:
        def weibull_MTBF(beta, eta):
            return eta * gamma(1 + 1/beta)

        P_fail = 1 - exp(- (t_fail/eta)**beta)
        MTBF = weibull_MTBF(beta, eta)
        riesgo = costo * P_fail

        # Control dinámico de área bajo la curva
        t = np.linspace(0, eta*2, 300)
        R = np.exp(-(t/eta)**beta)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(t, R, 'b-', lw=2, label='Confiabilidad R(t)')
        ax.fill_between(t, R, 0, where=(t<=t_fail), color='red', alpha=0.3, label='Área de probabilidad de fallo')
        ax.legend()
        ax.set_xlabel('Tiempo (días)')
        ax.set_ylabel('R(t)')
        ax.set_title('Evolución de la confiabilidad y probabilidad de fallo')
        ax.grid(True)
        st.pyplot(fig)

        st.latex(r"P(T \leq t) = 1 - e^{-(t/\eta)^{\beta}}")
        st.latex(r"E[T] = \eta \Gamma(1 + 1/\beta)")
        st.latex(r"Riesgo = Costo \times P(T \leq t)")

        st.markdown(f"**Probabilidad de fallo antes de {t_fail} días:** <span style='color:red; font-weight:bold;'>{P_fail:.3f}</span>", unsafe_allow_html=True)
        st.markdown(f"**MTBF (tiempo medio entre fallos):** <span style='color:blue; font-weight:bold;'>{MTBF:.2f} días</span>", unsafe_allow_html=True)
        st.markdown(f"**Riesgo económico estimado:** <span style='color:#F7B500; font-weight:bold;'>{riesgo:.0f} USD</span>", unsafe_allow_html=True)

        st.markdown("**Interpretación:** El área roja bajo la curva representa la probabilidad acumulada de fallo antes del tiempo t. En una planta láctea, este valor apoya la planificación de mantenimientos preventivos y la gestión del riesgo operativo.")

# =========================================================
# 3️⃣ MUESTRAS POSITIVAS EN LECHE CRUDA
# =========================================================
with tabs[2]:
    st.subheader("Muestras positivas en leche cruda — Probabilidad acumulada y periodo de retorno")

    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        p = st.slider("Probabilidad de positivo (p)", 0.001, 0.2, 0.03)
        n_max = st.slider("Semanas a evaluar", 5, 100, 40)

    with col2:
        n = np.arange(1, n_max+1)
        P_pos = 1 - (1 - p)**n
        T = 1 / p
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(n, P_pos, '-o', color='blue')
        ax2.set_xlabel("Semanas n")
        ax2.set_ylabel("Probabilidad acumulada")
        ax2.set_title("Probabilidad de detectar al menos un positivo")
        ax2.grid(True)
        st.pyplot(fig2)

        st.latex(r"T = \frac{1}{p}")
        st.latex(r"P(N \geq 1) = 1 - (1 - p)^n")
        st.markdown(f"**Periodo de retorno esperado:** <span style='color:blue; font-weight:bold;'>{T:.1f} semanas</span>", unsafe_allow_html=True)
        st.markdown("**Interpretación:** En control microbiológico, el periodo de retorno refleja la frecuencia esperada de detección de antibióticos. Un valor bajo alerta sobre recurrencia de contaminación.")

# =========================================================
# 4️⃣ INCERTIDUMBRE EN LA PROPORCIÓN (WILSON 95%)
# =========================================================
with tabs[3]:
    st.subheader("Incertidumbre en la proporción de positivos — Intervalo de Wilson 95%")

    col1, col2, col3 = st.columns([1.2, 2.2, 1.0])

    with col1:
        x = st.number_input("Número de positivos (x)", 0, 1000, 9)
        n_muestras = st.number_input("Número total de muestras (n)", 1, 10000, 300)

    with col2:
        z = 1.96
        p_hat = x / n_muestras
        num = p_hat + z**2/(2*n_muestras)
        den = 1 + z**2/n_muestras
        term = z * sqrt((p_hat*(1-p_hat)/n_muestras) + (z**2/(4*n_muestras**2)))
        IC_inf = (num - term) / den
        IC_sup = (num + term) / den

        p_vals = np.linspace(IC_inf, IC_sup, 200)
        T_vals = 1 / p_vals

        fig3, ax3 = plt.subplots(figsize=(7.5,4.5))
        ax3.plot(p_vals*100, T_vals, 'b-', lw=2)
        ax3.axvline(p_hat*100, color='k', ls='--')
        ax3.axvline(IC_inf*100, color='r', ls='--')
        ax3.axvline(IC_sup*100, color='r', ls='--')
        ax3.set_xlabel("Proporción de positivos (%)")
        ax3.set_ylabel("Periodo de retorno T = 1/p (semanas)")
        ax3.set_title("Incertidumbre en el periodo de retorno")
        ax3.grid(True)
        st.pyplot(fig3)

    with col3:
        st.latex(r"\hat{p} = \frac{x}{n}")
        st.latex(r"IC = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}")
        st.markdown(f"**Resultado:** <span style='color:blue; font-weight:bold;'>p̂ = {p_hat:.3f}, IC₉₅% ≈ [{IC_inf:.3f}, {IC_sup:.3f}]</span>", unsafe_allow_html=True)
        st.markdown("**Interpretación:** En el ámbito lácteo, este intervalo expresa la incertidumbre estadística de la proporción real de muestras contaminadas, útil para evaluar la eficacia de los controles de calidad.")

# Pie de página institucional
st.markdown("---")
st.markdown("<p style='text-align:center; color:#002F6C;'><b>M.Sc. Edwin Villarreal, Fís. — Universidad Politécnica Salesiana (UPS)</b></p>", unsafe_allow_html=True)


