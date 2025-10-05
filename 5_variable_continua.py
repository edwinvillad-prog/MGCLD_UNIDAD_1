# ==============================
# Importación de librerías
# ==============================
import numpy as np               # Para generar datos aleatorios y manejar arrays
import matplotlib.pyplot as plt  # Para graficar
import pandas as pd              # Para crear tabla de frecuencias
from scipy.stats import norm     # Para usar la distribución Normal teórica

# ==============================
# 1. Simulación de datos
# ==============================
# Genera 1000 mediciones de pH con distribución Normal
# Media (loc) = 4.5, Desviación estándar (scale) = 0.2
ph_muestras = np.random.normal(loc=4.5, scale=0.2, size=1000)

# ==============================
# 2. Tabla de frecuencias
# ==============================
# Calcula frecuencias absolutas por intervalos (10 bins)
conteos, bins = np.histogram(ph_muestras, bins=10)

# Construye un DataFrame con intervalos, frecuencias absolutas y relativas
tabla_frec = pd.DataFrame({
    "Intervalo": [f"{round(bins[i],2)} - {round(bins[i+1],2)}"
                  for i in range(len(bins)-1)],  # Rango de cada clase
    "Frecuencia": conteos,                       # Conteo de datos en el intervalo
    "Frecuencia Relativa": conteos / len(ph_muestras)  # Probabilidad empírica
})

# Imprime la tabla en consola
print("\nTabla de frecuencias (1000 muestras de pH):")
print(tabla_frec)

# ==============================
# 3. Histograma en frecuencias absolutas
# ==============================
plt.figure(figsize=(8,4))          # Tamaño de la figura
plt.hist(
    ph_muestras,                   # Datos simulados
    bins=10,                       # Número de intervalos
    color="lightgreen",            # Color de las barras
    edgecolor="black"              # Bordes de las barras
)

plt.title("Histograma en frecuencias absolutas (pH en yogur)")
plt.xlabel("pH")
plt.ylabel("Frecuencia")
plt.show()

# ==============================
# 4. Histograma normalizado + curva Normal teórica
# ==============================
plt.figure(figsize=(8,4))          # Nueva figura
plt.hist(
    ph_muestras,
    bins=15,                       # Más intervalos para suavizar
    color="lightgreen",
    edgecolor="black",
    density=True                   # Normaliza a densidad de probabilidad
)

# Crea un rango de valores de pH de 4.0 a 5.0
x = np.linspace(4.0, 5.0, 200)

# Calcula la densidad teórica de la Normal con media 4.5 y sigma 0.2
plt.plot(x, norm.pdf(x, 4.5, 0.2), 
         'r-', lw=2, label="Normal Teórica")

plt.title("Histograma normalizado vs Curva Normal")
plt.xlabel("pH")
plt.ylabel("Densidad")
plt.legend()
plt.show()
