# app_lacteos_maestria_final.py
# Laboratorio interactivo de Probabilidades para la Maestría — Industria Láctea
# Incluye:
# - Modo Clásico (General + Escenarios Lácteos) con TODOS los eventos y sombreado
# - Modo Avanzado (7 mejoras: LaTeX, interpretación, estadísticos, teoría vs simulación + ECM,
#   sensibilidad, comparación y exportación)

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import qrcode
import streamlit as st
from scipy.stats import binom, hypergeom, poisson, expon, norm, weibull_min
from streamlit_option_menu import option_menu

# =====================
# CONFIGURACIÓN
# =====================
st.set_page_config(page_title="Laboratorio de Probabilidades Lácteos", layout="wide")
st.title("🥛 Laboratorio de Probabilidades — Estadística Aplicada a la Industria Láctea")

# =====================
# HELPERS GENERALES
# =====================

def is_discrete_dist(dist):
    return hasattr(dist, "pmf")

def mostrar_formula(dist_name: str):
    """Muestra fórmulas en LaTeX para cada distribución"""
    fm = {
        "Binomial": r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}",
        "Hipergeométrica": r"P(X=k) = \dfrac{\binom{K}{k}\binom{M-K}{n-k}}{\binom{M}{n}}",
        "Poisson": r"P(X=k) = \dfrac{e^{-\lambda} \lambda^k}{k!}",
        "Exponencial": r"f(x) = \lambda e^{-\lambda x},\quad x\ge 0",
        "Normal": r"f(x) = \dfrac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
        "Weibull": r"f(x) = \dfrac{c}{\lambda} \left(\dfrac{x}{\lambda}\right)^{c-1} e^{-(x/\lambda)^c},\; x\ge 0",
    }
    if dist_name in fm:
        st.latex(fm[dist_name])


def estadisticos(dist):
    mean, var = dist.mean(), dist.var()
    q05, q50, q95 = dist.ppf([0.05, 0.5, 0.95])
    return pd.DataFrame({
        "Estadístico": ["Media", "Varianza", "Desviación estándar", "Q(0.05)", "Mediana", "Q(0.95)"],
        "Valor": [mean, var, np.sqrt(var), q05, q50, q95]
    })


def interpretar(prob: float, evento: str) -> str:
    return f"La probabilidad para **{evento}** es {prob:.4f}, es decir, ocurre en ~{100*prob:.2f}% de los casos."


def compute_prob(dist, tipo_evento: str, a: float, b: float | None):
    disc = is_discrete_dist(dist)
    if tipo_evento == "P(X = a)":
        return float(dist.pmf(a)) if disc else 0.0
    if tipo_evento == "P(X < a)":
        return float(dist.cdf(a-1)) if disc else float(dist.cdf(a))
    if tipo_evento == "P(X ≤ a)":
        return float(dist.cdf(a))
    if tipo_evento == "P(X > a)":
        return float(1 - dist.cdf(a))
    if tipo_evento == "P(X ≥ a)":
        return float(1 - (dist.cdf(a-1) if disc else dist.cdf(a)))
    if b is None:
        return np.nan
    if tipo_evento == "P(a < X < b)":
        return float((dist.cdf(b-1) - dist.cdf(a)) if disc else (dist.cdf(b) - dist.cdf(a)))
    if tipo_evento == "P(a ≤ X < b)":
        return float((dist.cdf(b-1) - dist.cdf(a-1)) if disc else (dist.cdf(b) - dist.cdf(a)))
    if tipo_evento == "P(a < X ≤ b)":
        return float((dist.cdf(b) - dist.cdf(a)) if disc else (dist.cdf(b) - dist.cdf(a)))
    if tipo_evento == "P(a ≤ X ≤ b)":
        return float((dist.cdf(b) - dist.cdf(a-1)) if disc else (dist.cdf(b) - dist.cdf(a)))
    return np.nan


def fig_pmf_pdf(dist, x_vals, tipo_evento: str, a: float, b: float | None, prob: float):
    disc = is_discrete_dist(dist)
    fig = go.Figure()
    if disc:
        y = dist.pmf(x_vals)
        fig.add_trace(go.Bar(x=x_vals, y=y, name="PMF", hovertemplate="x=%{x}<br>pmf=%{y:.4f}<extra></extra>"))
        # máscara sombreada
        mask = np.zeros_like(x_vals, dtype=bool)
        if tipo_evento == "P(X = a)": mask = (x_vals == a)
        elif tipo_evento == "P(X < a)": mask = (x_vals < a)
        elif tipo_evento == "P(X ≤ a)": mask = (x_vals <= a)
        elif tipo_evento == "P(X > a)": mask = (x_vals > a)
        elif tipo_evento == "P(X ≥ a)": mask = (x_vals >= a)
        elif b is not None:
            if tipo_evento == "P(a < X < b)": mask = (x_vals > a) & (x_vals < b)
            elif tipo_evento == "P(a ≤ X < b)": mask = (x_vals >= a) & (x_vals < b)
            elif tipo_evento == "P(a < X ≤ b)": mask = (x_vals > a) & (x_vals <= b)
            elif tipo_evento == "P(a ≤ X ≤ b)": mask = (x_vals >= a) & (x_vals <= b)
        fig.add_trace(go.Bar(
            x=x_vals[mask], y=y[mask], name=f"Área = {prob:.6f}", marker_color="crimson",
            hovertemplate=f"{tipo_evento}<br>P= {prob:.6f}<extra></extra>"
        ))
    else:
        y = dist.pdf(x_vals)
        fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name="PDF"))
        # sombreado continuo
        if tipo_evento in ("P(X = a)",):
            # nada que sombrear en continuas para prob puntual (0), pero marcamos línea vertical
            fig.add_vline(x=a, line_color="crimson")
        else:
            # rango para sombrear
            xa, xb = None, None
            if tipo_evento == "P(X < a)": xa, xb = x_vals.min(), a
            elif tipo_evento == "P(X ≤ a)": xa, xb = x_vals.min(), a
            elif tipo_evento == "P(X > a)": xa, xb = a, x_vals.max()
            elif tipo_evento == "P(X ≥ a)": xa, xb = a, x_vals.max()
            elif b is not None:
                if tipo_evento in ("P(a < X < b)", "P(a ≤ X < b)", "P(a < X ≤ b)", "P(a ≤ X ≤ b)"):
                    xa, xb = a, b
            if xa is not None and xb is not None and xb > xa:
                mask = (x_vals >= xa) & (x_vals <= xb)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_vals[mask], x_vals[mask][::-1]]),
                    y=np.concatenate([y[mask], np.zeros(mask.sum())]),
                    fill="toself", name=f"Área = {prob:.6f}", hoverinfo="skip",
                    fillcolor="rgba(220,20,60,0.35)", line=dict(color="rgba(0,0,0,0)")
                ))
                # anotación central con el valor exacto
                mid_x = (xa + xb) / 2
                fig.add_annotation(x=mid_x, y=max(y[mask]) if mask.any() else 0,
                                   text=f"P = {prob:.6f}", showarrow=False, bgcolor="rgba(255,255,255,0.6)")
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    return fig


def fig_cdf(dist, x_vals, a: float, b: float | None):
    disc = is_discrete_dist(dist)
    F = dist.cdf(x_vals)
    fig = go.Figure()
    if disc:
        fig.add_trace(go.Scatter(x=x_vals, y=F, mode="lines+markers", line_shape="hv", name="CDF"))
    else:
        fig.add_trace(go.Scatter(x=x_vals, y=F, mode="lines", name="CDF"))
    # marcas en a y b
    fig.add_vline(x=a, line_color="#888", line_dash="dot")
    if b is not None:
        fig.add_vline(x=b, line_color="#888", line_dash="dot")
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    return fig


def fig_simulacion(dist, x_vals, n_sim: int):
    disc = is_discrete_dist(dist)
    sample = dist.rvs(size=n_sim)
    fig = go.Figure()
    if disc:
        # histograma discreto con bins centrados en enteros
        bins = np.arange(x_vals.min()-0.5, x_vals.max()+1.5)
        fig.add_trace(go.Histogram(x=sample, xbins=dict(start=bins.min(), end=bins.max(), size=1),
                                   histnorm="probability", name="Simulación", opacity=0.6))
        y = dist.pmf(x_vals)
        fig.add_trace(go.Bar(x=x_vals, y=y, name="Teoría", opacity=0.6))
    else:
        fig.add_trace(go.Histogram(x=sample, nbinsx=40, histnorm="probability density",
                                   name="Simulación", opacity=0.5))
        y = dist.pdf(x_vals)
        fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name="Teoría"))
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    return fig, sample


def download_png(fig, filename: str):
    """Devuelve bytes PNG del fig si kaleido está disponible, si no muestra aviso."""
    try:
        img_bytes = fig.to_image(format="png", scale=2)  # requiere kaleido
        st.download_button("⬇️ Descargar gráfica (PNG)", data=img_bytes, file_name=filename, mime="image/png")
    except Exception:
        st.info("Para descargar como PNG instala `kaleido` (\n`pip install kaleido`) o usa el botón de cámara del gráfico.")

# =====================
# MODO PRINCIPAL
# =====================
modo_principal = st.radio("Seleccione el modo de trabajo:", ["Clásico", "Avanzado Maestría", "Comparar Distribuciones"], horizontal=True)

# =====================
# MODO CLÁSICO (completo)
# =====================
if modo_principal == "Clásico":
    submodo = st.radio("Submodo:", ["General", "Escenarios Lácteos"], horizontal=True)
    with st.sidebar:
        if submodo == "General":
            dist_name = option_menu(
                "Distribuciones",
                ["Binomial", "Hipergeométrica", "Poisson", "Exponencial", "Normal", "Weibull"],
                icons=["circle-fill", "square", "dot", "hourglass-split", "bar-chart", "triangle"],
                default_index=0, orientation="vertical")
        else:
            dist_name = option_menu(
                "Escenarios Lácteos",
                [
                    "Poisson — Conteo microbiológico",
                    "Normal — Contenido de grasa",
                    "Weibull — Vida útil yogur",
                    "Binomial — Defectos en yogures",
                    "Hipergeométrica — Control de lotes",
                    "Exponencial — Tiempo entre fallas",
                ],
                icons=["bug", "droplet", "hourglass", "box", "clipboard-check", "tools"],
                default_index=0, orientation="vertical")

    col1, col2 = st.columns(2)
    # Parámetros (General)
    if submodo == "General":
        if dist_name == "Binomial":
            n = col1.number_input("n (ensayos)", 1, 500, 10)
            p = col2.slider("p (éxito)", 0.0, 1.0, 0.5)
            dist, x_vals = binom(n, p), np.arange(0, n+1)
        elif dist_name == "Hipergeométrica":
            M = col1.number_input("M (población)", 1, 2000, 100)
            K = col2.number_input("K (éxitos en población)", 0, M, 10)
            n = st.number_input("n (muestra)", 1, M, 5)
            dist, x_vals = hypergeom(M, K, n), np.arange(0, n+1)
        elif dist_name == "Poisson":
            mu = col1.number_input("λ (media)", 0.1, 200.0, 5.0)
            max_x = int(max(20, mu + 10*np.sqrt(mu)))
            dist, x_vals = poisson(mu), np.arange(0, max_x)
        elif dist_name == "Exponencial":
            lam = col1.number_input("λ (tasa)", 0.01, 10.0, 0.2)
            dist, x_vals = expon(scale=1/lam), np.linspace(0, 10/lam, 400)
        elif dist_name == "Normal":
            mu = col1.number_input("mu (media)", min_value=-1000.0, max_value=1000.0, value=0.0, step=0.1)
            sigma = col2.number_input("sigma (desviacion)", min_value=0.001, max_value=1000.0, value=1.0, step=0.001)
            dist, x_vals = norm(mu, sigma), np.linspace(mu - 4*sigma, mu + 4*sigma, 600)
        elif dist_name == "Weibull":
            c = col1.number_input("Forma (c)", 0.1, 10.0, 1.5)
            scale = col2.number_input("Escala (λ)", 0.01, 1e3, 10.0)
            dist, x_vals = weibull_min(c, scale=scale), np.linspace(0, 5*scale, 600)
    else:
        # Escenarios Lácteos
        if "Poisson" in dist_name:
            dist, x_vals = poisson(5), np.arange(0, 30)
            st.info("🧫 Conteo de UFC en placa (λ=5)")
        elif "Normal" in dist_name:
            dist, x_vals = norm(3.5, 0.2), np.linspace(2.5, 4.5, 600)
            st.info("🥛 Contenido de grasa en leche (μ=3.5%, σ=0.2)")
        elif "Weibull" in dist_name:
            dist, x_vals = weibull_min(1.5, scale=10), np.linspace(0, 60, 600)
            st.info("🍶 Vida útil de yogur (c=1.5, λ=10 días)")
        elif "Binomial" in dist_name:
            dist, x_vals = binom(20, 0.1), np.arange(0, 21)
            st.info("🍦 Defectos en yogures (n=20, p=0.1)")
        elif "Hipergeométrica" in dist_name:
            dist, x_vals = hypergeom(100, 10, 5), np.arange(0, 6)
            st.info("📦 Control de lotes (M=100, K=10, n=5)")
        elif "Exponencial" in dist_name:
            dist, x_vals = expon(scale=5), np.linspace(0, 60, 600)
            st.info("⚙️ Tiempo entre fallas (λ=0.2 → media=5 h)")

    # Eventos
    st.subheader("Eventos a calcular")
    tipo_evento = st.selectbox(
        "Seleccione el evento",
        [
            "P(X = a)", "P(X < a)", "P(X ≤ a)", "P(X > a)", "P(X ≥ a)",
            "P(a < X < b)", "P(a ≤ X < b)", "P(a < X ≤ b)", "P(a ≤ X ≤ b)",
        ],
        index=6  # por defecto un intervalo común
    )
    cols_ev = st.columns(2)
    a = cols_ev[0].number_input("a:", value=1.0)
    b = None
    if "b)" in tipo_evento or "≤ b)" in tipo_evento or "< b)" in tipo_evento:
        b = cols_ev[1].number_input("b:", value=2.0)
    if b is not None and b <= a and ("< b" in tipo_evento or "≤ b" in tipo_evento):
        st.warning("Para los intervalos, asegúrate de que b > a.")

    # Cálculo exacto
    prob = compute_prob(dist, tipo_evento, a, b)
    st.success(f"Probabilidad: {prob:.6f}")

    # Gráficas con sombreado
    t1, t2, t3 = st.tabs(["PMF/PDF", "CDF", "Simulación"])
    with t1:
        fig = fig_pmf_pdf(dist, x_vals, tipo_evento, a, b, prob)
        st.plotly_chart(fig, use_container_width=True)
        download_png(fig, "pmf_pdf.png")
    with t2:
        fig_c = fig_cdf(dist, x_vals, a, b)
        st.plotly_chart(fig_c, use_container_width=True)
        download_png(fig_c, "cdf.png")
    with t3:
        n_sim = st.slider("Número de simulaciones", 100, 20000, 2000, step=100)
        fig_s, sample = fig_simulacion(dist, x_vals, n_sim)
        st.plotly_chart(fig_s, use_container_width=True)
        download_png(fig_s, "simulacion.png")

        # Tabla resumida y descarga CSV
        if is_discrete_dist(dist):
            y_pdf = dist.pmf(x_vals)
        else:
            y_pdf = dist.pdf(x_vals)
        df_out = pd.DataFrame({"x": x_vals, "PDF/PMF": y_pdf, "CDF": dist.cdf(x_vals)})
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV (teoría)", data=csv, file_name="teoria.csv", mime="text/csv")

# =====================
# MODO AVANZADO (7 mejoras)
# =====================
elif modo_principal == "Avanzado Maestría":
    with st.sidebar:
        dist_name = option_menu(
            "Distribución",
            ["Binomial", "Hipergeométrica", "Poisson", "Exponencial", "Normal", "Weibull"],
            icons=["circle-fill", "square", "dot", "hourglass-split", "bar-chart", "triangle"],
            default_index=2, orientation="vertical")

    st.subheader("Formulación matemática (LaTeX)")
    mostrar_formula(dist_name)

    # Parámetros + sensibilidad (sliders)
    col1, col2 = st.columns(2)
    if dist_name == "Binomial":
        n = col1.slider("n (ensayos)", 1, 1000, 20)
        p = col2.slider("p (éxito)", 0.0, 1.0, 0.3)
        dist, x_vals = binom(n, p), np.arange(0, n+1)
    elif dist_name == "Hipergeométrica":
        M = col1.slider("M (población)", 10, 5000, 200)
        K = col2.slider("K (éxitos en población)", 0, M, 20)
        n = st.slider("n (muestra)", 1, M, 30)
        dist, x_vals = hypergeom(M, K, n), np.arange(0, n+1)
    elif dist_name == "Poisson":
        lam = col1.slider("λ (media)", 0.1, 200.0, 5.0)
        max_x = int(max(20, lam + 10*np.sqrt(lam)))
        dist, x_vals = poisson(lam), np.arange(0, max_x)
    elif dist_name == "Exponencial":
        lam = col1.slider("λ (tasa)", 0.01, 10.0, 0.2)
        dist, x_vals = expon(scale=1/lam), np.linspace(0, 10/lam, 600)
    elif dist_name == "Normal":
        mu = col1.slider("μ (media)", -50.0, 50.0, 0.0)
        sigma = col2.slider("σ (desviación)", 0.01, 20.0, 1.0)
        dist, x_vals = norm(mu, sigma), np.linspace(mu-4*sigma, mu+4*sigma, 600)
    elif dist_name == "Weibull":
        c = col1.slider("Forma (c)", 0.1, 10.0, 1.5)
        scale = col2.slider("Escala (λ)", 0.01, 200.0, 10.0)
        dist, x_vals = weibull_min(c, scale=scale), np.linspace(0, 5*scale, 600)

    # Estadísticos
    st.subheader("📊 Estadísticos de la distribución")
    st.dataframe(estadisticos(dist), use_container_width=True)

    # Eventos (con interpretación)
    st.subheader("Eventos e interpretación")
    tipo_evento = st.selectbox(
        "Evento",
        [
            "P(X = a)", "P(X < a)", "P(X ≤ a)", "P(X > a)", "P(X ≥ a)",
            "P(a < X < b)", "P(a ≤ X < b)", "P(a < X ≤ b)", "P(a ≤ X ≤ b)",
        ], index=6)
    ca, cb = st.columns(2)
    a = ca.number_input("a:", value=1.0)
    b = None
    if "b)" in tipo_evento or "≤ b)" in tipo_evento or "< b)" in tipo_evento:
        b = cb.number_input("b:", value=2.0)
    prob = compute_prob(dist, tipo_evento, a, b)
    st.success(f"Probabilidad: {prob:.6f}")
    st.info(interpretar(prob, tipo_evento))

    # Gráficos y simulación con ECM
    t1, t2, t3 = st.tabs(["PMF/PDF (sensibilidad)", "CDF", "Simulación vs Teoría + ECM"])
    with t1:
        fig = fig_pmf_pdf(dist, x_vals, tipo_evento, a, b, prob)
        st.plotly_chart(fig, use_container_width=True)
        download_png(fig, "pmf_pdf_avanzado.png")
    with t2:
        figc = fig_cdf(dist, x_vals, a, b)
        st.plotly_chart(figc, use_container_width=True)
        download_png(figc, "cdf_avanzado.png")
    with t3:
        n_sim = st.slider("Simulaciones", 100, 50000, 5000, step=100)
        fig_s, sample = fig_simulacion(dist, x_vals, n_sim)
        st.plotly_chart(fig_s, use_container_width=True)
        # ECM (aproxima con el mismo binning usado en fig_simulacion)
        if is_discrete_dist(dist):
            theo = dist.pmf(x_vals)
            # histograma discreto
            counts = np.bincount(sample.astype(int), minlength=len(x_vals))
            emp = counts / counts.sum()
            ecm = float(np.mean((emp - theo)**2))
        else:
            # continuo: densidad en bins
            hist, bins = np.histogram(sample, bins=40, density=True)
            centers = 0.5*(bins[:-1]+bins[1:])
            theo = dist.pdf(centers)
            # normalizar escalas aproximando
            theo = theo / np.trapz(theo, centers)
            ecm = float(np.mean((hist - theo)**2))
        st.write(f"⚡ ECM teoría–simulación: **{ecm:.6f}**")
        download_png(fig_s, "sim_vs_teo_avanzado.png")

    # Exportación
    y_pdf = dist.pmf(x_vals) if is_discrete_dist(dist) else dist.pdf(x_vals)
    df = pd.DataFrame({"x": x_vals, "PDF/PMF": y_pdf, "CDF": dist.cdf(x_vals)})
    st.download_button("⬇️ Descargar resultados (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="resultados_avanzado.csv", mime="text/csv")

# =====================
# COMPARAR DISTRIBUCIONES (parametrizable)
# =====================
elif modo_principal == "Comparar Distribuciones":
    st.info("Compare dos distribuciones (parámetros configurables). Útil para ver aproximaciones (p.ej., Binomial → Poisson).")
    cols = st.columns(2)
    tipos = ["Normal", "Poisson", "Binomial", "Exponencial", "Weibull"]

    with cols[0]:
        d1 = st.selectbox("Distribución 1", tipos, index=0, key="d1")
        if d1 == "Normal":
            mu1 = st.number_input("μ1", value=0.0, key="mu1")
            s1 = st.number_input("σ1", value=1.0, min_value=0.01, key="s1")
            dist1 = norm(mu1, s1); x1 = np.linspace(mu1-4*s1, mu1+4*s1, 600)
        elif d1 == "Poisson":
            l1 = st.number_input("λ1", value=5.0, min_value=0.1, key="l1")
            max1 = int(max(20, l1 + 10*np.sqrt(l1)))
            dist1 = poisson(l1); x1 = np.arange(0, max1)
        elif d1 == "Binomial":
            n1 = st.number_input("n1", value=20, min_value=1, key="n1")
            p1 = st.number_input("p1", value=0.3, min_value=0.0, max_value=1.0, key="p1")
            dist1 = binom(n1, p1); x1 = np.arange(0, n1+1)
        elif d1 == "Exponencial":
            l1 = st.number_input("λ1 (tasa)", value=0.2, min_value=0.01, key="l1e")
            dist1 = expon(scale=1/l1); x1 = np.linspace(0, 10/l1, 600)
        elif d1 == "Weibull":
            c1 = st.number_input("c1 (forma)", value=1.5, min_value=0.1, key="c1")
            sc1 = st.number_input("λ1 (escala)", value=10.0, min_value=0.01, key="sc1")
            dist1 = weibull_min(c1, scale=sc1); x1 = np.linspace(0, 5*sc1, 600)

    with cols[1]:
        d2 = st.selectbox("Distribución 2", tipos, index=1, key="d2")
        if d2 == "Normal":
            mu2 = st.number_input("μ2", value=0.0, key="mu2")
            s2 = st.number_input("σ2", value=2.0, min_value=0.01, key="s2")
            dist2 = norm(mu2, s2); x2 = np.linspace(mu2-4*s2, mu2+4*s2, 600)
        elif d2 == "Poisson":
            l2 = st.number_input("λ2", value=7.0, min_value=0.1, key="l2")
            max2 = int(max(20, l2 + 10*np.sqrt(l2)))
            dist2 = poisson(l2); x2 = np.arange(0, max2)
        elif d2 == "Binomial":
            n2 = st.number_input("n2", value=30, min_value=1, key="n2")
            p2 = st.number_input("p2", value=0.2, min_value=0.0, max_value=1.0, key="p2")
            dist2 = binom(n2, p2); x2 = np.arange(0, n2+1)
        elif d2 == "Exponencial":
            l2 = st.number_input("λ2 (tasa)", value=0.5, min_value=0.01, key="l2e")
            dist2 = expon(scale=1/l2); x2 = np.linspace(0, 10/l2, 600)
        elif d2 == "Weibull":
            c2 = st.number_input("c2 (forma)", value=2.0, min_value=0.1, key="c2")
            sc2 = st.number_input("λ2 (escala)", value=8.0, min_value=0.01, key="sc2")
            dist2 = weibull_min(c2, scale=sc2); x2 = np.linspace(0, 5*sc2, 600)

    # Dominio combinado
    disc1, disc2 = is_discrete_dist(dist1), is_discrete_dist(dist2)
    x_min = min(x1.min(), x2.min())
    x_max = max(x1.max(), x2.max())
    x = np.linspace(x_min, x_max, 800) if not (disc1 and disc2) else np.arange(int(np.floor(x_min)), int(np.ceil(x_max))+1)

    fig = go.Figure()
    # D1
    if disc1:
        fig.add_trace(go.Bar(x=x1, y=dist1.pmf(x1), name=f"{d1}", opacity=0.5))
    else:
        fig.add_trace(go.Scatter(x=x, y=dist1.pdf(x) if hasattr(dist1, "pdf") else dist1.pmf(x),
                                 mode="lines", name=f"{d1}"))
    # D2
    if disc2:
        fig.add_trace(go.Bar(x=x2, y=dist2.pmf(x2), name=f"{d2}", opacity=0.5))
    else:
        fig.add_trace(go.Scatter(x=x, y=dist2.pdf(x) if hasattr(dist2, "pdf") else dist2.pmf(x),
                                 mode="lines", name=f"{d2}"))

    fig.update_layout(barmode="overlay", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")  # reemplaza use_container_width=True
    download_png(fig, "comparacion.png")

from PIL import Image

url = "https://mystreamlitapp.streamlit.app"
qr = qrcode.make(url)

# Guardar el código QR en un archivo PNG local
qr.save("qr_app.png")

print("✅ Código QR generado y guardado como 'qr_app.png'")
