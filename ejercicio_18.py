import numpy as np
from scipy import stats

# Datos de las 5 réplicas del precio por barril de petróleo
precios = np.array([125.50, 132.75, 120.80, 138.20, 127.50])
n = len(precios)

# Estadísticos muestrales
media = np.mean(precios)
s = np.std(precios, ddof=1)

print("=" * 75)
print("EJERCICIO 18: DETERMINACIÓN DE EXACTITUD Y NÚMERO DE RÉPLICAS")
print("=" * 75)
print(f"Número inicial de réplicas (n): {n}")
print(f"Media muestral (x_bar): ${media:.2f} USD/barril")
print(f"Desviación estándar muestral (s): ${s:.4f} USD/barril")
print("-" * 75)

# -------------------------------------------------------------------------
# Inciso a) Exactitud lograda con 5 réplicas al 95% de confianza (alpha = 0.05)
# -------------------------------------------------------------------------
alpha_a = 0.05
gl_a = n - 1
t_crit_a = stats.t.ppf(1 - alpha_a / 2, df=gl_a)
exactitud_a = t_crit_a * (s / np.sqrt(n))
ic_inf_a = media - exactitud_a
ic_sup_a = media + exactitud_a

print("a) EXACTITUD LOGRADA (CONFIANZA 95%):")
print(f"   * Grados de libertad: {gl_a}")
print(f"   * t crítico (t_0.025, 4): {t_crit_a:.4f}")
print(f"   * Exactitud (semi-ancho h): ±${exactitud_a:.4f} USD/barril")
print(f"   * Intervalo de confianza: [${ic_inf_a:.2f}, ${ic_sup_a:.2f}] USD/barril")
print("-" * 75)

# -------------------------------------------------------------------------
# Inciso b) Número de réplicas requeridas para exactitud h = ±0.35 al 90%
# -------------------------------------------------------------------------
h_deseada = 0.35
alpha_b = 0.10

# Proceso iterativo usando la distribución t de Student
n_estimado = 5
while True:
    t_crit_b = stats.t.ppf(1 - alpha_b / 2, df=n_estimado - 1)
    n_siguiente = int(np.ceil((t_crit_b * s / h_deseada) ** 2))
    if n_siguiente <= n_estimado:
        break
    n_estimado = n_siguiente

print("b) NÚMERO DE RÉPLICAS REQUERIDAS (EXACTITUD ±0.35, CONFIANZA 90%):")
print(f"   * Margen de error tolerado (h): ±${h_deseada:.2f} USD/barril")
print(f"   * Nivel de significancia (alpha): {alpha_b:.2f}")
print(f"   * Valor t crítico final utilizado: {t_crit_b:.4f}")
print(f"   * Total de réplicas necesarias (n'): {n_estimado} réplicas")
print(f"   * Réplicas adicionales a ejecutar: {n_estimado - n} réplicas")
print("=" * 75)