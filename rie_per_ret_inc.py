import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import exp, sqrt

# ------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------
st.set_page_config(page_title="Riesgo, Periodo de Retorno e Incertidumbre en Procesos Lácteos", page_icon="🧮", layout="wide")

# === Colores institucionales UPS ===
UPS_BLUE = "#002F6C"
UPS_GOLD = "#F7B500"
UPS_RED  = "#D32F2F"
UPS_BG   = "#F8FAFF"

st.markdown(f"""
<h1 style='text-align:center; color:{UPS_BLUE};'>🧮 Riesgo, Periodo de Retorno e Incertidumbre en Procesos Lácteos</h1>
<h4 style='text-align:center; color:{UPS_GOLD};'>Universidad Politécnica Salesiana — Posgrados UPS</h4>
<hr style='border:2px solid {UPS_BLUE};'>
""", unsafe_allow_html=True)

# =========================================================
# PESTAÑAS PRINCIPALES
# =========================================================
tabs = st.tabs(["📊 Fallos de Equipos", "⚙️ Fallo en Bomba CIP", "🥛 Presencia de Contaminantes", "📈 Incertidumbre"])

# =========================================================
# 1️⃣ MODELO DE FALLOS WEIBULL
# =========================================================
with tabs[0]:
    st.subheader("Modelo de fallos Weibull — Confiabilidad de equipos lácteos")
    st.markdown(f"<h5 style='color:{UPS_GOLD};'>Análisis del comportamiento de fallos en válvulas, bombas CIP y equipos de proceso</h5>", unsafe_allow_html=True)
    st.markdown("El modelo Weibull permite caracterizar los patrones de fallos de equipos industriales a lo largo del tiempo, diferenciando entre fallas tempranas, aleatorias y por desgaste.")

    col_panel, col_plot = st.columns([1.2, 1.8])

    with col_panel:
        st.markdown("### Parámetros e interpretación en la industria láctea")
        st.markdown("**β (forma):** Controla el tipo de fallo. En bombas o válvulas: **β<1** → fallas tempranas (defectos); **β≈1** → aleatorias; **β>1** → desgaste progresivo.")
        st.markdown("**η (escala):** Vida característica (≈ tiempo en el que falla el 63% de las unidades).")
        st.markdown("**t (tiempo):** Horizonte de operación (días).")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            beta1 = st.slider("β₁ (Tempranas, β<1)", 0.20, 0.99, 0.90, help="Controla la tasa de fallas iniciales. Valores menores a 1 implican defectos de fabricación o instalación.")
            beta3 = st.slider("β₂ (Desgaste, β>1)", 1.01, 3.00, 1.50, help="Representa fallas progresivas por envejecimiento de componentes o fricción.")
            beta2 = st.slider("β₃ (Aleatorias, ≈1)", 0.90, 1.10, 1.00, help="Modela fallos aleatorios, asociados a eventos no predecibles.")
        with col2:
            eta1 = st.slider("η₁ (Tempranas, días)", 5, 50, 20, help="Vida característica del equipo con fallas tempranas.")
            eta2 = st.slider("η₂ (Desgaste, días)", 5, 50, 25, help="Tiempo medio de vida útil antes del desgaste significativo.")
            eta3 = st.slider("η₃ (Aleatorias, días)", 5, 50, 12, help="Tiempo medio de vida bajo condiciones aleatorias.")
        t_max = st.slider("Tiempo máximo (días)", 10, 120, 40)

    with col_plot:
        def weibull_R(t, beta, eta):
            return np.exp(-(t/eta)**beta)

        t = np.linspace(0, t_max, 400)
        fig, ax = plt.subplots(figsize=(6,4))
        for beta, eta, label, color in zip(
            [beta1, beta2, beta3],
            [eta1, eta2, eta3],
            ["Tempranas (β<1)", "Desgaste (β>1)", "Aleatorias (β≈1)"],
            [UPS_RED, UPS_BLUE, UPS_GOLD]):
            ax.plot(t, weibull_R(t, beta, eta), lw=2, label=f"{label}: β={beta:.2f}, η={eta:.1f}", color=color)
        ax.set_xlabel("t (días)")
        ax.set_ylabel("R(t)")
        ax.set_title("Curvas de Confiabilidad Weibull")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.latex(r"R(t) = e^{-(t/\eta)^{\beta}}")
        st.caption("Donde R(t) representa la probabilidad de que el equipo continúe operativo tras t días; η es la vida característica y β define el tipo de fallo.")

        st.markdown("### Resumen de parámetros ingresados")
        st.table({
            'Tipo de falla': ['Tempranas', 'Desgaste', 'Aleatorias'],
            'β (forma)': [beta1, beta2, beta3],
            'η (escala, días)': [eta1, eta2, eta3]
        })

# =========================================================
# 2️⃣ EJEMPLO: BOMBA CIP
# =========================================================
with tabs[1]:
    st.subheader("Ejemplo: Bomba CIP — Evaluación del riesgo de fallos")
    st.markdown(f"<h5 style='color:{UPS_GOLD};'>Cálculo de la confiabilidad, MTBF y riesgo económico</h5>", unsafe_allow_html=True)
    st.markdown("Este módulo estima la probabilidad de fallo, el tiempo medio hasta el fallo (MTBF) y el riesgo económico esperado para una bomba CIP.")

    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.markdown("### Parámetros de entrada e interpretación")
        st.markdown("**β:** patrón de fallas (β>1 → desgaste del impulsor/sellos).")
        st.markdown("**η:** vida media característica (días).")
        st.markdown("**t:** tiempo operativo o revisión (días).")
        st.markdown("**Costo:** pérdida económica por parada no planificada (USD).")

        beta = st.number_input("β (forma)", 0.10, 5.00, 1.50)
        eta = st.number_input("η (escala, días)", 1.0, 120.0, 25.0)
        costo = st.number_input("Costo por parada (USD)", 0.0, 20000.0, 1200.0)
        t_fail = st.slider("Tiempo de evaluación t (días)", 1, 120, 20)

    with col2:
        def weibull_MTBF(beta, eta):
            return eta * gamma(1 + 1/beta)

        P_fail = 1 - exp(- (t_fail/eta)**beta)
        MTBF = weibull_MTBF(beta, eta)
        riesgo = costo * P_fail

        t = np.linspace(0, int(eta*2), 400)
        R = np.exp(-(t/eta)**beta)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(t, R, 'b-', lw=2, label='Confiabilidad R(t)')
        ax.fill_between(t, R, 0, where=(t<=t_fail), color=UPS_RED, alpha=0.3, label='P(fallo antes de t)')
        ax.axvline(t_fail, color='k', ls='--', lw=1)
        ax.legend()
        ax.set_xlabel('Tiempo (días)')
        ax.set_ylabel('R(t)')
        ax.set_title('Confiabilidad y probabilidad acumulada de fallo')
        ax.grid(True)
        st.pyplot(fig)

        st.latex(r"P(T \leq t) = 1 - e^{-(t/\eta)^{\beta}}")
        st.caption("Probabilidad acumulada de fallo antes del tiempo t.")
        st.latex(r"E[T] = \eta \, \Gamma\!\left(1 + \frac{1}{\beta}\right)")
        st.caption("MTBF: tiempo medio esperado hasta el fallo.")
        st.latex(r"\text{Riesgo esperado} = Costo \times P(T \leq t)")
        st.caption("Valor económico esperado asociado al fallo dentro del intervalo analizado.")

        st.markdown(f"""
        <div style='background:{UPS_BG};border-left:4px solid {UPS_BLUE};padding:10px;border-radius:6px;'>
        <b>Conclusiones:</b><br>
        • <b>Probabilidad de fallo antes de t:</b> <span style='color:{UPS_RED};font-weight:700'>{P_fail:.3f}</span><br>
        • <b>MTBF (tiempo medio hasta el fallo):</b> <span style='color:{UPS_BLUE};font-weight:700'>{MTBF:.2f} días</span><br>
        • <b>Riesgo económico esperado:</b> <span style='color:#F57C00;font-weight:700'>{riesgo:.0f} USD</span><br>
        <i>Interpretación láctea:</i> El área roja cuantifica la fracción de bombas que fallarán antes de t días; útil para programar mantenimientos preventivos.
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# 3️⃣ MUESTRAS POSITIVAS EN LECHE CRUDA
# =========================================================
with tabs[2]:
    st.subheader("Muestras positivas en leche cruda — Probabilidad acumulada y periodo de retorno")
    st.markdown(f"<h5 style='color:{UPS_GOLD};'>Modelado de la detección de contaminantes y estimación de frecuencia esperada</h5>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        p = st.slider("Probabilidad de positivo semanal p", 0.001, 0.2, 0.03, help="Proporción esperada de muestras positivas por antibióticos.")
        n_max = st.slider("Semanas a visualizar (curva)", 5, 100, 40, help="Duración del monitoreo en semanas.")
        n_eval = st.number_input("Semanas de interés n (cálculo puntual)", 1, 1000, 20, help="Número de semanas para el cálculo puntual.")

    with col2:
        n = np.arange(1, n_max+1)
        P_curve = 1 - (1 - p)**n
        T = 1 / p
        P_puntual = 1 - (1 - p)**n_eval

        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(n, P_curve, '-o', color=UPS_BLUE)
        ax2.axvline(n_eval, color='k', ls='--')
        ax2.axhline(P_puntual, color='k', ls='--')
        ax2.set_xlabel("Semanas n")
        ax2.set_ylabel("Probabilidad acumulada")
        ax2.set_title("Probabilidad de detectar al menos un positivo")
        ax2.grid(True)
        st.pyplot(fig2)

        st.latex(r"T = \frac{1}{p}")
        st.caption("Periodo medio entre positivos consecutivos (frecuencia esperada).")
        st.latex(r"P(N \geq 1) = 1 - (1 - p)^n")
        st.caption("Probabilidad de detectar al menos un caso positivo en n semanas.")

        st.markdown(f"""
        <div style='background:{UPS_BG};border-left:4px solid {UPS_BLUE};padding:10px;border-radius:6px;'>
        <b>Resultados:</b><br>
        • <b>Periodo de retorno esperado T:</b> <span style='color:{UPS_BLUE};font-weight:700'>{T:.1f} semanas</span><br>
        • <b>Probabilidad de al menos un positivo en n={n_eval} semanas:</b> <span style='color:#2E7D32;font-weight:700'>{P_puntual:.1%}</span><br>
        <i>Interpretación láctea:</i> T indica cada cuántas semanas, en promedio, aparece un positivo por antibióticos. Valores bajos advierten sobre recurrencia de contaminación.
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# 4️⃣ INCERTIDUMBRE EN LA PROPORCIÓN (WILSON 95%)
# =========================================================
with tabs[3]:
    st.subheader("Incertidumbre en la proporción de positivos — Intervalo de Wilson 95%")
    st.markdown(f"<h5 style='color:{UPS_GOLD};'>Evaluación de la precisión estadística de la proporción y del periodo de retorno</h5>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 2.2])
    with col1:
        x = st.number_input("Número de positivos (x)", 0, 100000, 9, help="Número total de muestras con resultados positivos.")
        n_muestras = st.number_input("Número total de muestras (n)", 1, 1000000, 300, help="Tamaño total de la muestra analizada.")

    with col2:
        z = 1.96
        p_hat = x / n_muestras
        num = p_hat + z**2/(2*n_muestras)
        den = 1 + z**2/n_muestras
        term = z * sqrt((p_hat*(1-p_hat)/n_muestras) + (z**2/(4*n_muestras**2)))
        IC_inf = (num - term) / den
        IC_sup = (num + term) / den

        T_inf = 1 / IC_sup
        T_sup = 1 / IC_inf

        p_vals = np.linspace(max(IC_inf,1e-6), IC_sup, 200)
        T_vals = 1 / p_vals

        fig3, ax3 = plt.subplots(figsize=(7.8,4.8))
        ax3.plot(p_vals*100, T_vals, 'b-', lw=2)
        ax3.axvline(p_hat*100, color='k', ls='--')
        ax3.axvline(IC_inf*100, color='r', ls='--')
        ax3.axvline(IC_sup*100, color='r', ls='--')
        ax3.set_xlabel("Proporción de positivos (%)")
        ax3.set_ylabel("Periodo de retorno T = 1/p (semanas)")
        ax3.set_title("Incertidumbre en el periodo de retorno")
        ax3.grid(True)
        st.pyplot(fig3)

        st.latex(r"\hat{p} = \frac{x}{n}")
        st.caption("Proporción puntual de positivos detectados.")
        st.latex(r"IC_{95\%}(p) = \frac{\hat{p} + \frac{z^2}{2n} \pm z\,\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}")
        st.caption("Intervalo de confianza de Wilson (95%) para la proporción real de positivos.")
        st.latex(r"IC_{95\%}(T=1/p) = \left[\;\frac{1}{p_{\text{sup}}}\;,\; \frac{1}{p_{\text{inf}}}\;\right]")
        st.caption("Intervalo de confianza transformado al dominio del periodo de retorno (T).")

        st.markdown(f"""
        <div style='background:{UPS_BG};border-left:4px solid #880E4F;padding:10px;border-radius:6px;'>
        <b>Resultados:</b><br>
        • <b>p̂:</b> <span style='font-weight:700'>{p_hat:.3f}</span><br>
        • <b>IC₉₅%(p):</b> <span style='font-weight:700'>[{IC_inf:.3f}, {IC_sup:.3f}]</span><br>
        • <b>IC₉₅%(T = 1/p):</b> <span style='font-weight:700'>[{T_inf:.1f}, {T_sup:.1f}] semanas</span><br>
        <i>Interpretación láctea:</i> El intervalo de T muestra el rango probable de semanas entre positivos; útil para definir frecuencia de monitoreo y control de calidad.
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------
# PIE DE PÁGINA INSTITUCIONAL
# ------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:{UPS_BLUE};'><b>M.Sc. Edwin Villarreal, Fís. — Universidad Politécnica Salesiana (UPS)</b></p>", unsafe_allow_html=True)
