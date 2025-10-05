# Ejemplo Teorema de Bayes en yogures
def bayes_yogur():
    # Probabilidades previas
    P_correcto = 0.95
    P_defectuoso = 0.05
    # Probabilidades condicionales
    P_acidez_correcto = 0.01
    P_acidez_defectuoso = 0.20
    # Probabilidad total de acidez fuera de rango
    P_acidez_total = (P_correcto * P_acidez_correcto +
    P_defectuoso * P_acidez_defectuoso)
    # Aplicacin de Bayes
    P_defectuoso_given_acidez = (
    P_defectuoso * P_acidez_defectuoso) / P_acidez_total
    print("Probabilidad de defecto dado acidez:",
    round(P_defectuoso_given_acidez, 3))
bayes_yogur()