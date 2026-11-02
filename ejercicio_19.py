import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main():
    n_muestras = 300
    seed = 42
    np.random.seed(seed)

    # Parámetros de x1: Normal(mu=30, var=0.81 -> sigma=0.9)
    mu_1 = 30.0
    sigma_1 = np.sqrt(0.81)

    # Parámetros de x2: Erlang(k=10, media=14.5 -> theta=1.45)
    k_erlang = 10
    media_2 = 14.5
    theta_erlang = media_2 / k_erlang

    # Generación de variables aleatorias
    x1 = np.random.normal(loc=mu_1, scale=sigma_1, size=n_muestras)
    # Una Erlang(k, theta) se genera como Gamma(shape=k, scale=theta)
    x2 = np.random.gamma(shape=k_erlang, scale=theta_erlang, size=n_muestras)

    # Longitud de la barra unida
    longitud_total = x1 + x2

    # Límites de especificación
    LIE = 45.0 - 5.0  # 40.0 cm
    LSE = 45.0 + 5.0  # 50.0 cm

    # Conteo de barras fuera de especificación
    fuera_especificaciones = (longitud_total < LIE) | (longitud_total > LSE)
    n_defectuosas = int(np.sum(fuera_especificaciones))
    prob_estimada = n_defectuosas / n_muestras

    # Cálculo de media y desviación estándar de la soldadura
    media_y = float(np.mean(longitud_total))
    s_y = float(np.std(longitud_total, ddof=1))

    # Cálculo de los índices Cp y Cpk
    Cp = (LSE - LIE) / (6.0 * s_y)
    Cpi = (media_y - LIE) / (3.0 * s_y)
    Cps = (LSE - media_y) / (3.0 * s_y)
    Cpk = min(Cpi, Cps)

    print("=" * 75)
    print("EJERCICIO 19: SIMULACIÓN DE SOLDADURA DE 300 BARRAS")
    print("=" * 75)
    print(f"Número de barras simuladas: {n_muestras}")
    print(f"Media observada (Y): {media_y:.4f} cm (Teórica: 44.5000 cm)")
    print(f"Desviación estándar (s): {s_y:.4f} cm (Teórica: 4.6728 cm)")
    print(f"Límites de diseño: [{LIE:.1f} cm , {LSE:.1f} cm]")
    print("-" * 75)
    print(f"Barras fuera de especificaciones: {n_defectuosas} de {n_muestras}")
    print(f"Estimador de la probabilidad P(Fuera de esp.): {prob_estimada:.4f} ({prob_estimada*100:.2f}%)")
    print("-" * 75)
    print(f"ÍNDICES DE CAPACIDAD DEL PROCESO:")
    print(f"  * Cp  = {Cp:.4f}")
    print(f"  * Cpi = {Cpi:.4f}")
    print(f"  * Cps = {Cps:.4f}")
    print(f"  * Cpk = {Cpk:.4f}")
    print("=" * 75)

    # Gráfico del proceso vs especificaciones
    plt.figure(figsize=(10, 5))
    conteo, bins, _ = plt.hist(longitud_total, bins=25, color="#3498db", edgecolor="black", alpha=0.7, density=True, label="Barras simuladas")
    plt.axvline(LIE, color="red", linestyle="--", linewidth=2, label=f"LIE = {LIE:.1f} cm")
    plt.axvline(LSE, color="red", linestyle="--", linewidth=2, label=f"LSE = {LSE:.1f} cm")
    plt.axvline(media_y, color="green", linestyle="-", linewidth=2, label=f"Media = {media_y:.2f} cm")

    plt.title("Distribución de Longitud de Barras Soldadas vs Especificaciones", fontsize=13)
    plt.xlabel("Longitud Total (cm)", fontsize=11)
    plt.ylabel("Densidad", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("soldadura_barras_ejercicio_19.png", dpi=300)
    print("Gráfico generado como 'soldadura_barras_ejercicio_19.png'.")
    plt.show()

if __name__ == "__main__":
    main()