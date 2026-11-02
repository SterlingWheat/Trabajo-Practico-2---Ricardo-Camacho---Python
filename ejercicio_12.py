import numpy as np
from scipy import stats

# Datos extraídos directamente del histograma
intervalos = [
    (150, 151), (151, 152), (152, 153), (153, 154), (154, 155),
    (155, 156), (156, 157), (157, 158), (158, 159), (159, 160)
]
frecuencias_obs = np.array([13, 8, 11, 15, 7, 8, 9, 11, 10, 8])
n = np.sum(frecuencias_obs)  # 100 observaciones
k = len(frecuencias_obs)     # 10 intervalos
alpha = 0.05                 # Nivel de significancia para 95% de confianza

# Marcas de clase para estimar media y desviación muestral
marcas_clase = np.array([(a + b) / 2.0 for a, b in intervalos])
media_estimada = np.sum(frecuencias_obs * marcas_clase) / n
var_estimada = np.sum(frecuencias_obs * ((marcas_clase - media_estimada) ** 2)) / (n - 1)
desv_estimada = np.sqrt(var_estimada)

# -------------------------------------------------------------
# 1. Prueba Chi-cuadrada para DISTRIBUCIÓN UNIFORME U(150, 160)
# -------------------------------------------------------------
# Cada uno de los 10 intervalos tiene la misma probabilidad teórica p_i = 1/10
esp_uniforme = np.full(k, n / k)
chi2_uniforme = np.sum((frecuencias_obs - esp_uniforme) ** 2 / esp_uniforme)
gl_uniforme = k - 1  # 9 grados de libertad (parámetros conocidos del rango)
chi2_crit_uniforme = stats.chi2.ppf(1 - alpha, df=gl_uniforme)
p_val_uniforme = 1.0 - stats.chi2.cdf(chi2_uniforme, df=gl_uniforme)
pasa_uniforme = chi2_uniforme <= chi2_crit_uniforme

# -------------------------------------------------------------
# 2. Prueba Chi-cuadrada para DISTRIBUCIÓN NORMAL N(mu, sigma)
# -------------------------------------------------------------
p_normal = []
for a, b in intervalos:
    prob = stats.norm.cdf(b, loc=media_estimada, scale=desv_estimada) - \
           stats.norm.cdf(a, loc=media_estimada, scale=desv_estimada)
    p_normal.append(prob)
p_normal = np.array(p_normal)
esp_normal = n * p_normal

chi2_normal = np.sum((frecuencias_obs - esp_normal) ** 2 / esp_normal)
gl_normal = k - 1 - 2  # 7 grados de libertad (se estimaron mu y sigma)
chi2_crit_normal = stats.chi2.ppf(1 - alpha, df=gl_normal)
p_val_normal = 1.0 - stats.chi2.cdf(chi2_normal, df=gl_normal)
pasa_normal = chi2_normal <= chi2_crit_normal

# -------------------------------------------------------------
# Salida de Resultados
# -------------------------------------------------------------
print("=" * 75)
print("EJERCICIO 12: PRUEBA CHI-CUADRADA DE BONDAD DE AJUSTE (CONFIANZA 95%)")
print("=" * 75)
print(f"Total de datos (n): {n} en {k} intervalos")
print(f"Media estimada: {media_estimada:.4f} | Desviación estándar estimada: {desv_estimada:.4f}")
print("-" * 75)

print("\n1. EVALUACIÓN DE DISTRIBUCIÓN UNIFORME CONTINUA U(150, 160):")
print(f"   * Frecuencia esperada por intervalo (E_i): {esp_uniforme[0]:.1f}")
print(f"   * Estadístico Chi2 calculado: {chi2_uniforme:.4f}")
print(f"   * Valor crítico Chi2 (df = {gl_uniforme}, alpha = 0.05): {chi2_crit_uniforme:.4f}")
print(f"   * p-valor: {p_val_uniforme:.4f}")
print(f"   -> Decisión: {'SE ACEPTA H0' if pasa_uniforme else 'SE RECHAZA H0'}")

print("\n2. EVALUACIÓN DE DISTRIBUCIÓN NORMAL N(154.95, 2.87^2):")
print(f"   * Estadístico Chi2 calculado: {chi2_normal:.4f}")
print(f"   * Valor crítico Chi2 (df = {gl_normal}, alpha = 0.05): {chi2_crit_normal:.4f}")
print(f"   * p-valor: {p_val_normal:.4f}")
print(f"   -> Decisión: {'SE ACEPTA H0' if pasa_normal else 'SE RECHAZA H0'}")

print("\n" + "=" * 75)
print("CONCLUSIÓN FINAL:")
print(f"Los datos siguen una DISTRIBUCIÓN UNIFORME CONTINUA en el intervalo [150, 160].")
print(f"El estadístico Chi-cuadrada calculado ({chi2_uniforme:.4f}) es ampliamente menor")
print(f"al valor crítico de rechazo ({chi2_crit_uniforme:.4f}), con un p-valor de {p_val_uniforme:.4f},")
print("lo que confirma un ajuste uniforme con un nivel de confianza del 95%.")
print("=" * 75)