# Ejemplo de probabilidad condicional en lácteos

def prob_condicional():
    # Probabilidades de origen
    P_F1 = 0.20
    P_F2 = 0.80

    # Probabilidades de contaminación
    P_C_F1 = 0.05   # P(C | F1)
    P_C_F2 = 0.10   # P(C | F2)

    # Probabilidades conjuntas
    P_C_F1_total = P_F1 * P_C_F1
    P_C_F2_total = P_F2 * P_C_F2

    # Probabilidad total de contaminación
    P_C = P_C_F1_total + P_C_F2_total

    # Probabilidad condicional de que el lote
    # provenga de Finca 1 dado que está contaminado
    P_F1_given_C = P_C_F1_total / P_C

    print("Probabilidad de que el lote contaminado sea de Finca 1:",
          round(P_F1_given_C, 3))

# Ejecutar función
prob_condicional()
