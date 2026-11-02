import numpy as np
import pandas as pd
from scipy import stats

# 1. 200 Datos históricos de la demanda diaria por televisor
DEMANDA_HIST = [
    99, 94, 80, 73, 86, 95, 90, 70, 92, 83, 89, 93, 62, 94, 86, 62, 77, 72, 63, 88, 96,
    99, 98, 77, 96, 74, 86, 81, 94, 81, 70, 98, 78, 86, 74, 94, 95, 77, 79, 89, 65, 94,
    88, 72, 71, 70, 76, 95, 92, 96, 93, 72, 97, 75, 97, 72, 76, 69, 61, 69, 99, 83, 83,
    61, 64, 95, 80, 70, 79, 65, 80, 85, 72, 68, 83, 78, 84, 64, 72, 67, 94, 100, 78, 93,
    66, 99, 97, 74, 93, 95, 70, 93, 64, 86, 72, 80, 83, 65, 66, 96, 95, 84, 88, 68, 65,
    70, 88, 90, 85, 92, 81, 68, 92, 64, 84, 69, 73, 71, 98, 66, 68, 74, 93, 94, 86, 69,
    82, 71, 79, 100, 84, 65, 99, 100, 61, 78, 73, 80, 72, 100, 100, 98, 82, 60, 93, 75,
    84, 75, 76, 71, 94, 74, 84, 72, 71, 71, 85, 69, 88, 97, 93, 63, 91, 89, 89, 99, 93,
    62, 99, 77, 69, 82, 66, 91, 73, 88, 61, 63, 67, 76, 100, 93, 82, 72, 75, 92, 87, 74,
    60, 93, 77, 60, 84, 64, 83, 75, 68, 64, 97, 78
]

# 2. Datos históricos de la cantidad de productos inspeccionados por el cliente al día (68 registros)
INSPECCIONES_HIST = [
    4, 2, 3, 0, 0, 2, 4, 1, 2, 0, 2, 2, 3, 3, 2, 1, 1, 1, 4, 2, 1, 0, 3, 1, 2,
    3, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 3, 1, 0, 3, 0, 1, 0, 1,
    2, 0, 1, 2, 0, 1, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1, 1, 3, 1, 0, 0, 1, 1, 2,
    0, 0, 0, 1, 0, 2, 0, 2, 1, 2, 0, 1, 1, 2, 0, 1, 2, 1, 2, 3, 0, 0, 0, 2, 2
]

def simular_politica(Q, dias=1000, seed=42):
    """
    Simula una política de producción fija Q televisores/día durante un horizonte dado.
    """
    rng = np.random.default_rng(seed)
    costos_diarios = []

    # Muestreo de demandas e inspecciones para los días simulados
    demandas = rng.choice(DEMANDA_HIST, size=dias, replace=True)
    inspecciones = rng.choice(INSPECCIONES_HIST, size=dias, replace=True)

    # Defectos internos en planta ~ Binomial(Q, 0.03)
    defectos_internos = rng.binomial(n=Q, p=0.03, size=dias)
    
    # Defectos encontrados por el cliente ~ Binomial(N_insp, 0.05)
    defectos_cliente = rng.binomial(n=inspecciones, p=0.05, size=dias)

    for t in range(dias):
        q_aptos = Q - defectos_internos[t]
        demanda_t = demandas[t]

        # Costos por faltante e inventario
        if q_aptos >= demanda_t:
            faltante = 0
            sobrante = q_aptos - demanda_t
        else:
            faltante = demanda_t - q_aptos
            sobrante = 0

        costo_faltante = faltante * 40.0
        costo_inventario = sobrante * 20.0
        costo_def_interno = defectos_internos[t] * 90.0
        costo_def_cliente = defectos_cliente[t] * 120.0

        costo_dia = costo_faltante + costo_inventario + costo_def_interno + costo_def_cliente
        costos_diarios.append(costo_dia)

    costos_diarios = np.array(costos_diarios)
    media = float(np.mean(costos_diarios))
    s = float(np.std(costos_diarios, ddof=1))
    
    # Intervalo de confianza al 95%
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, df=dias - 1)
    error_std = s / np.sqrt(dias)
    ic_inf = media - t_crit * error_std
    ic_sup = media + t_crit * error_std

    return {
        "Q": Q,
        "media": media,
        "s": s,
        "ic_inf": ic_inf,
        "ic_sup": ic_sup
    }

def main():
    politicas = [60, 70, 80, 90, 100]
    resultados = []

    print("=" * 85)
    print("EJERCICIO 22: DETERMINACIÓN DE POLÍTICA ÓPTIMA DE PRODUCCIÓN DE TELEVISORES")
    print("=" * 85)

    for q in politicas:
        res = simular_politica(Q=q, dias=5000, seed=123)
        resultados.append(res)

    df_res = pd.DataFrame(resultados)
    
    print("\nRESUMEN DE COSTOS PROMEDIO POR DÍA E INTERVALOS DE CONFIANZA (95%):")
    print("-" * 85)
    print(f"{'Política (Q)':15} | {'Costo Medio ($/día)':20} | {'Desv. Est. (s)':15} | {'IC 95% ($/día)':25}")
    print("-" * 85)
    for r in resultados:
        print(f"Q = {r['Q']:<11} | ${r['media']:>16.2f} | ${r['s']:>11.2f} | [${r['ic_inf']:.2f}, ${r['ic_sup']:.2f}]")
    print("-" * 85)

    # Identificar la política de menor costo
    mejor = min(resultados, key=lambda x: x["media"])
    print(f"\nPOLÍTICA RECOMENDADA: Producir Q = {mejor['Q']} televisores por día.")
    print(f"Costo promedio mínimo esperado: ${mejor['media']:.2f} USD/día.")
    print("=" * 85)

if __name__ == "__main__":
    main()