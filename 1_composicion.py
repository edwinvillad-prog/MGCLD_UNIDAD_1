import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def analisis_composicion_leche():
    """
    Resuelve el problema de análisis de composición
    """
    # Probabilidades individuales
    P_grasa = 0.85    # P(contenido de grasa adecuado)
    P_proteina = 0.90 # P(contenido de proteína adecuado)
    P_lactosa = 0.80  # P(contenido de lactosa adecuado)

    # Probabilidad de muestra completa
    P_completa = P_grasa * P_proteina * P_lactosa

    # Parte 1: Probabilidad de que las 3 muestras sean completas
    P_tres_completas = P_completa ** 3
    print(f"1. P(3 completas): {P_tres_completas:.6f}")

    # Parte 2: Al menos una muestra completa
    P_ninguna_completa = (1 - P_completa) ** 3
    P_al_menos_una = 1 - P_ninguna_completa
    print(f"2. P(al menos 1): {P_al_menos_una:.6f}")

    # Parte 3: Exactamente 2 con grasa adecuada
    n = 3
    k = 2
    P_exactamente_2_grasa = binom.pmf(k, n, P_grasa)
    print(f"3. P(exactamente 2 grasa): {P_exactamente_2_grasa:.6f}")

    return P_tres_completas, P_al_menos_una, P_exactamente_2_grasa


# === Ejecutar análisis ===
if __name__ == "__main__":
    prob_analiticas = analisis_composicion_leche()
