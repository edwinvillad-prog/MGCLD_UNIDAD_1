import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binom

# Parámetros de la Binomial
n_lote = 50          # Tamaño del lote (50 quesos)
p_defecto = 0.1      # Probabilidad de que un queso sea defectuoso (10%)

# Simulación de 1000 lotes
muestras = np.random.binomial(n_lote, p_defecto, 1000)

# ==============================
# TABLA DE FRECUENCIAS
# ==============================
valores, conteos = np.unique(muestras, return_counts=True)

# Crear DataFrame con frecuencias absolutas y relativas
tabla_frec = pd.DataFrame({
    "Defectuosos": valores,
    "Frecuencia": conteos,
    "Frecuencia Relativa": conteos / len(muestras)
})

print("\nTabla de frecuencias (simulación 1000 lotes):")
print(tabla_frec)

# ==============================
# GRÁFICA 1: FRECUENCIAS ABSOLUTAS
# ==============================
plt.figure(figsize=(8,4))
plt.hist(
    muestras,
    bins=range(0, n_lote+2),
    color="lightblue",
    edgecolor="black",
    align="left"
)
plt.title("Histograma en frecuencias absolutas")
plt.xlabel("Número de defectuosos")
plt.ylabel("Frecuencia")
plt.show()

# ==============================
# GRÁFICA 2: PROBABILIDADES + CURVA BINOMIAL
# ==============================
plt.figure(figsize=(8,4))
plt.hist(
    muestras,
    bins=range(0, n_lote+2),
    color="skyblue",
    edgecolor="black",
    align="left",
    density=True
)

# Curva teórica de la Binomial
x = np.arange(0, n_lote+1)
plt.plot(x, binom.pmf(x, n_lote, p_defecto),
         'ro-', lw=2, label="Teoría Binomial")

plt.title("Histograma de probabilidades vs Teoría Binomial")
plt.xlabel("Número de defectuosos")
plt.ylabel("Probabilidad")
plt.legend()
plt.show()

