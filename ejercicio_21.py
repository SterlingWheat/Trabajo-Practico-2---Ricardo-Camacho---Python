import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def simular_tiempo_reparacion():
    """Genera el tipo de falla y su respectivo tiempo de reparación en horas."""
    # Ruleta de probabilidades según frecuencias históricas (total = 85 días)
    r_tipo = random.random()
    p_leve = 10 / 85
    p_moderada = 15 / 85
    p_grave = 25 / 85

    if r_tipo < p_leve:
        # 4-Erlang(media = 4) -> suma de 4 exponenciales de media 1
        # t = -theta * ln(prod(r_i)) con k=4, theta=1.0
        prod_r = (
            random.random() * random.random() * random.random() * random.random()
        )
        t = -1.0 * math.log(prod_r)
    elif r_tipo < p_leve + p_moderada:
        # Normal(mu = 3.5, sigma = 1.1)
        t = random.gauss(3.5, 1.1)
        t = max(0.0, t)
    elif r_tipo < p_leve + p_moderada + p_grave:
        # f(t) = t - 6 en [6, 6 + sqrt(2)] -> t = 6 + sqrt(2 * r)
        r = random.random()
        t = 6.0 + math.sqrt(2.0 * r)
    else:
        # Exponencial con lambda = 0.2 (media = 5 horas)
        r = random.random()
        t = -5.0 * math.log(1.0 - r)

    # Acotamos a un máximo de 24 horas por día
    return min(24.0, t)


def main():
    dias = 300
    random.seed(42)
    np.random.seed(42)

    producciones = []
    utilidades = []
    tiempos_paro = []

    for _ in range(dias):
        # 1. Evaluar si la máquina se descompone (probabilidad 0.80)
        se_descompone = random.random() < 0.80

        if se_descompone:
            t_rep = simular_tiempo_reparacion()
        else:
            t_rep = 0.0

        # 2. Producción diaria (disminuye proporcionalmente a la reparación)
        fraccion_operativa = (24.0 - t_rep) / 24.0
        piezas = 1000.0 * fraccion_operativa

        # 3. Utilidad diaria: Ventas ($100/pieza) - Costo de reparación ($11/hora)
        ingresos = piezas * 100.0
        costos = t_rep * 11.0
        utilidad = ingresos - costos

        producciones.append(piezas)
        utilidades.append(utilidad)
        tiempos_paro.append(t_rep)

    producciones = np.array(producciones)
    utilidades = np.array(utilidades)
    tiempos_paro = np.array(tiempos_paro)

    # Promedios acumulados
    dias_eje = np.arange(1, dias + 1)
    prom_prod_acum = np.cumsum(producciones) / dias_eje
    prom_util_acum = np.cumsum(utilidades) / dias_eje

    # Métricas finales
    prod_media = float(np.mean(producciones))
    util_media = float(np.mean(utilidades))
    t_paro_medio = float(np.mean(tiempos_paro))

    # Intervalo de confianza al 95% para la utilidad media diaria
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, df=dias - 1)
    err_util = stats.sem(utilidades)
    ic_util_inf = util_media - t_crit * err_util
    ic_util_sup = util_media + t_crit * err_util

    print("=" * 75)
    print("EJERCICIO 21: SIMULACIÓN DE MÁQUINA DE PRODUCCIÓN (300 DÍAS)")
    print("=" * 75)
    print(f"Días evaluados: {dias} días (24 horas cada uno)")
    print(f"Tiempo promedio diario de inactividad/reparación: {t_paro_medio:.2f} horas")
    print("-" * 75)
    print("a) ÍNDICE DE PRODUCCIÓN DIARIA:")
    print(f"   * Producción promedio diaria tras 300 días: {prod_media:.2f} piezas/día")
    print(f"   * Pérdida promedio por paros: {1000.0 - prod_media:.2f} piezas/día")
    print("-" * 75)
    print("b) UTILIDAD DIARIA:")
    print(f"   * Utilidad promedio diaria tras 300 días: ${util_media:,.2f} USD/día")
    print(f"   * Intervalo de Confianza al 95%: [${ic_util_inf:,.2f}, ${ic_util_sup:,.2f}] USD/día")
    print("=" * 75)

    # Gráficas de estabilización
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(dias_eje, prom_prod_acum, color="#2980b9", linewidth=2, label="Producción diaria promedio acumulada")
    ax1.axhline(prod_media, color="#c0392b", linestyle="--", label=f"Media final = {prod_media:.1f} piezas")
    ax1.set_ylabel("Piezas / día", fontsize=11)
    ax1.set_title("Estabilización de la Producción Promedio Diaria", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.plot(dias_eje, prom_util_acum, color="#27ae60", linewidth=2, label="Utilidad diaria promedio acumulada")
    ax2.axhline(util_media, color="#c0392b", linestyle="--", label=f"Media final = ${util_media:,.2f}")
    ax2.set_xlabel("Días de Operación", fontsize=11)
    ax2.set_ylabel("Utilidad ($ / día)", fontsize=11)
    ax2.set_title("Estabilización de la Utilidad Promedio Diaria", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("produccion_utilidad_ejercicio_21.png", dpi=300)
    print("\nGráfico exportado como 'produccion_utilidad_ejercicio_21.png'.")
    plt.show()


if __name__ == "__main__":
    main()