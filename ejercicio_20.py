import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simular_un_dia(duracion_minutos=480.0):
    """
    Simula una jornada bancaria de 8 horas (480 minutos) con 1 cajera y 1 fila.
    Retorna el tiempo de espera promedio en fila de los clientes atendidos ese día.
    """
    tiempo_actual = 0.0
    fin_servicio_anterior = 0.0
    tiempos_en_fila = []

    while True:
        # Tiempo entre llegadas ~ Exp(media = 4.0 minutos) -> 15 personas/hora
        inter_arribo = -4.0 * math.log(1.0 - random.random())
        tiempo_llegada = tiempo_actual + inter_arribo

        # La puerta cierra a las 8 horas (480 minutos)
        if tiempo_llegada > duracion_minutos:
            break

        tiempo_actual = tiempo_llegada

        # Determinar número de transacciones según probabilidades dadas
        r_trans = random.random()
        if r_trans < 0.40:
            k_trans = 1
        elif r_trans < 0.70:
            k_trans = 2
        elif r_trans < 0.90:
            k_trans = 3
        else:
            k_trans = 4

        # Tiempo total de servicio del cliente: suma de k transacciones Exp(media = 2.2 min)
        tiempo_servicio = 0.0
        for _ in range(k_trans):
            tiempo_servicio += -2.2 * math.log(1.0 - random.random())

        # Dinámica de la cola FIFO con un servidor
        inicio_servicio = max(tiempo_llegada, fin_servicio_anterior)
        tiempo_espera = inicio_servicio - tiempo_llegada
        tiempos_en_fila.append(tiempo_espera)

        fin_servicio_anterior = inicio_servicio + tiempo_servicio

    if len(tiempos_en_fila) == 0:
        return 0.0

    return float(np.mean(tiempos_en_fila))


def main():
    dias = 300
    random.seed(42)
    np.random.seed(42)

    promedios_por_dia = []

    print("=" * 80)
    print("EJERCICIO 20: SIMULACIÓN DE BANCO (1 CAJERA, 8 HORAS/DÍA, 300 RÉPLICAS)")
    print("=" * 80)
    print("Simulando los 300 días de operación...")

    for dia in range(dias):
        prom_dia = simular_un_dia(duracion_minutos=480.0)
        promedios_por_dia.append(prom_dia)

    promedios_por_dia = np.array(promedios_por_dia)
    promedio_global = float(np.mean(promedios_por_dia))
    desv_entre_dias = float(np.std(promedios_por_dia, ddof=1))

    # Intervalo de confianza al 95% para la media global (n = 300 días)
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, df=dias - 1)
    error_estandar = desv_entre_dias / np.sqrt(dias)
    ic_inf = promedio_global - t_crit * error_estandar
    ic_sup = promedio_global + t_crit * error_estandar

    print(f"\nRESULTADOS ESTADÍSTICOS TRAS 300 DÍAS DE RÉPLICA:")
    print("-" * 80)
    print(f"Número de días simulados: {dias} jornadas de 8 horas")
    print(f"Tiempo promedio en fila: {promedio_global:.2f} minutos ({promedio_global/60:.2f} horas)")
    print(f"Desviación estándar diaria: {desv_entre_dias:.2f} minutos")
    print(f"Día con menor tiempo promedio en fila: {np.min(promedios_por_dia):.2f} minutos")
    print(f"Día con mayor tiempo promedio en fila: {np.max(promedios_por_dia):.2f} minutos")
    print(f"Intervalo de Confianza al 95%: [{ic_inf:.2f}, {ic_sup:.2f}] minutos")
    print("=" * 80)

    # Gráfico de convergencia del promedio acumulado a lo largo de los 300 días
    dias_eje = np.arange(1, dias + 1)
    prom_acumulado = np.cumsum(promedios_por_dia) / dias_eje

    plt.figure(figsize=(12, 6))
    plt.plot(dias_eje, prom_acumulado, color="#2980b9", linewidth=2, label="Tiempo promedio acumulado en fila")
    plt.axhline(y=promedio_global, color="#e74c3c", linestyle="--", linewidth=1.5,
                label=f"Media Global Final = {promedio_global:.2f} min")
    plt.title("Convergencia del Tiempo Promedio en Fila a lo largo de 300 Días Simulados", fontsize=13)
    plt.xlabel("Días de Operación (Réplicas)", fontsize=11)
    plt.ylabel("Tiempo Promedio en Fila (minutos)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("banco_cola_ejercicio_20.png", dpi=300)
    print("Gráfico exportado exitosamente como 'banco_cola_ejercicio_20.png'.")
    plt.show()


if __name__ == "__main__":
    main()