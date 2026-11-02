import math
import random
import matplotlib.pyplot as plt
import numpy as np


def generar_datos(n=5000, seed=42):
    random.seed(seed)

    # Listas para almacenar los valores individuales
    fa_vals = []
    fb_vals = []
    fc_vals = []

    for _ in range(n):
        # Función a: -20 * ln(ri * rj)
        ri = random.random()
        rj = random.random()
        xa = -20.0 * math.log(ri * rj)
        fa_vals.append(xa)

        # Función b: -10 * ln(ri * rj * rk * rl)
        prod_b = (
            random.random() * random.random() * random.random() * random.random()
        )
        xb = -10.0 * math.log(prod_b)
        fb_vals.append(xb)

        # Función c: -5 * ln(ri * ... * rp) (8 variables)
        prod_c = 1.0
        for _ in range(8):
            prod_c *= random.random()
        xc = -5.0 * math.log(prod_c)
        fc_vals.append(xc)

    return np.array(fa_vals), np.array(fb_vals), np.array(fc_vals)


def main():
    N = 10000  # Número total de mantenimientos simulados
    fa, fb, fc = generar_datos(n=N, seed=2026)

    # Cálculo del promedio acumulado conforme se realizan mantenimientos
    mantenimientos = np.arange(1, N + 1)
    prom_a = np.cumsum(fa) / mantenimientos
    prom_b = np.cumsum(fb) / mantenimientos
    prom_c = np.cumsum(fc) / mantenimientos

    # Graficar convergencia
    plt.figure(figsize=(12, 6))
    plt.plot(
        mantenimientos, prom_a, label="Función a: -20*ln(2 vars)", color="#e74c3c", alpha=0.8
    )
    plt.plot(
        mantenimientos, prom_b, label="Función b: -10*ln(4 vars)", color="#f39c12", alpha=0.8
    )
    plt.plot(
        mantenimientos, prom_c, label="Función c: -5*ln(8 vars)", color="#27ae60", alpha=0.8
    )

    # Valor esperado teórico
    plt.axhline(
        y=40.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Valor esperado teórico (40.0)",
    )

    # Banda de estabilidad de ±2% (39.2 a 40.8)
    plt.axhspan(39.2, 40.8, color="gray", alpha=0.15, label="Banda de estabilidad (±2%)")

    plt.title(
        "Convergencia del Promedio de Tiempos de Reparación (Estado Estable)",
        fontsize=14,
    )
    plt.xlabel("Número de Mantenimientos Realizados", fontsize=12)
    plt.ylabel("Tiempo Promedio de Reparación", fontsize=12)
    plt.ylim(25, 55)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Guardar gráfico y mostrarlo
    plt.savefig("convergencia_ejercicio_04.png", dpi=300)
    print("Gráfico exportado exitosamente como 'convergencia_ejercicio_04.png'.")
    plt.show()

    # Análisis del punto de convergencia (entrar en banda ±2% y permanecer allí)
    def punto_estabilidad(promedios, meta=40.0, tolerancia=0.02):
        lim_inf = meta * (1 - tolerancia)
        lim_sup = meta * (1 + tolerancia)
        fuera_de_banda = np.where((promedios < lim_inf) | (promedios > lim_sup))[0]
        if len(fuera_de_banda) == 0:
            return 1
        return fuera_de_banda[-1] + 2

    estab_a = punto_estabilidad(prom_a)
    estab_b = punto_estabilidad(prom_b)
    estab_c = punto_estabilidad(prom_c)

    print("\n" + "=" * 50)
    print("RESULTADOS DEL ANÁLISIS DE ESTADO ESTABLE:")
    print("=" * 50)
    print(f"Valor al que converge el promedio en (a): {prom_a[-1]:.2f}")
    print(f"Valor al que converge el promedio en (b): {prom_b[-1]:.2f}")
    print(f"Valor al que converge el promedio en (c): {prom_c[-1]:.2f}")
    print("-" * 50)
    print(f"Mantenimientos para estabilizarse (criterio ±2%):")
    print(f" - Función a: {estab_a} mantenimientos")
    print(f" - Función b: {estab_b} mantenimientos")
    print(f" - Función c: {estab_c} mantenimientos")
    print("=" * 50)


if __name__ == "__main__":
    main()