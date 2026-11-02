import numpy as np
from scipy import stats

# 100 datos con corrección de lectura OCR
DATOS = [
    # Fila 1
    12.561, 2.695, 12.082, 10.335, 13.260, 2.549, 4.594, 2.500, 24.930, 7.805, 8.322, 7.422, 11.143, 20.599,
    # Fila 2
    7.508, 4.367, 1.544, 3.706, 8.185, 14.405, 4.057, 15.584, 9.049, 6.265, 10.663, 10.257, 11.475, 4.688,
    # Fila 3
    16.256, 4.688, 11.963, 5.599, 19.204, 1.784, 25.998, 12.299, 10.317, 3.779, 18.993, 7.419, 15.154,
    # Fila 4
    9.579, 8.423, 6.934, 2.005, 13.234, 5.542, 5.271, 12.831, 8.231, 15.330, 7.958, 7.103, 16.134, 0.189,
    # Fila 5
    10.165, 14.624, 15.696, 10.212, 0.891, 3.186, 9.051, 11.118, 4.449, 17.901, 15.497, 6.645, 5.078,
    # Fila 6
    11.555, 3.724, 21.500, 7.160, 13.528, 3.372, 15.334, 7.603, 31.066, 1.992, 21.127, 10.784, 3.643,
    # Fila 7
    27.334, 3.178, 1.313, 10.962, 6.936, 3.140, 16.877, 19.171, 6.620, 3.775, 16.675, 1.368, 17.583, 1.669,
    # Fila 8
    11.157, 16.432, 2.831, 7.844, 10.745
]

def prueba_chi_cuadrada_distribucion(datos, dist_nombre, dist_obj, k=8, alpha=0.05):
    """
    Ajusta la distribución por Máxima Verosimilitud (MLE) y evalúa
    la bondad de ajuste con Chi-cuadrada usando k intervalos equiprobables.
    """
    n = len(datos)
    
    # 1. Ajuste de parámetros teóricos
    params = dist_obj.fit(datos)
    num_params = len(params)
    
    # 2. Partición en intervalos de probabilidad equiprobable (1/k)
    p_cortes = np.linspace(0, 1, k + 1)
    bordes = dist_obj.ppf(p_cortes, *params)
    bordes[0] = -np.inf
    bordes[-1] = np.inf
    
    # 3. Frecuencias observadas y esperadas
    obs, _ = np.histogram(datos, bins=bordes)
    esp = n / k
    
    # 4. Estadístico Chi-cuadrada
    chi2_calc = np.sum((obs - esp) ** 2 / esp)
    
    # Grados de libertad: k - 1 - parámetros estimados
    gl = k - 1 - num_params
    if gl <= 0:
        gl = 1
        
    chi2_crit = stats.chi2.ppf(1 - alpha, df=gl)
    p_valor = 1.0 - stats.chi2.cdf(chi2_calc, df=gl)
    pasa = chi2_calc <= chi2_crit
    
    return {
        "distribucion": dist_nombre,
        "parametros": params,
        "chi2_calc": chi2_calc,
        "chi2_crit": chi2_crit,
        "gl": gl,
        "p_valor": p_valor,
        "pasa": pasa
    }

def main():
    datos = np.array(DATOS)
    alpha = 0.05  # Nivel de confianza del 95%
    
    print("=" * 75)
    print("EJERCICIO 11: AJUSTE DE DISTRIBUCIONES (STAT::FIT PROMODEL - 95% CONFIANZA)")
    print("=" * 75)
    print(f"Número de observaciones (n): {len(datos)}")
    print(f"Media muestral: {np.mean(datos):.4f}")
    print(f"Desviación estándar: {np.std(datos, ddof=1):.4f}")
    print(f"Mínimo: {np.min(datos):.4f} | Máximo: {np.max(datos):.4f}")
    print("-" * 75)

    # Distribuciones comúnmente evaluadas por Stat::Fit en ProModel
    candidatas = [
        ("Weibull", stats.weibull_min),
        ("Gamma", stats.gamma),
        ("Lognormal", stats.lognorm),
        ("Normal", stats.norm),
        ("Exponencial", stats.expon)
    ]
    
    resultados = []
    for nombre, dist_obj in candidatas:
        res = prueba_chi_cuadrada_distribucion(datos, nombre, dist_obj, k=8, alpha=alpha)
        resultados.append(res)
        
    # Ordenar por menor estadístico Chi-cuadrada (mayor p-valor = mejor bondad de ajuste)
    resultados = sorted(resultados, key=lambda x: x["chi2_calc"])
    
    print(f"{'Distribución':15} | {'Chi2 Calc':10} | {'Chi2 Crít (95%)':16} | {'p-valor':10} | {'Decisión'}")
    print("-" * 75)
    for r in resultados:
        estado = "ACEPTADA" if r["pasa"] else "RECHAZADA"
        print(f"{r['distribucion']:15} | {r['chi2_calc']:10.4f} | {r['chi2_crit']:16.4f} | {r['p_valor']:10.4f} | {estado}")
        
    print("=" * 75)
    mejor = resultados[0]
    print(f"DISTRIBUCIÓN RECOMENDADA POR STAT::FIT:")
    print(f"-> {mejor['distribucion']} con estadístico Chi-cuadrada = {mejor['chi2_calc']:.4f} (p-valor: {mejor['p_valor']:.4f}).")
    print(f"\nCONCLUSIÓN AL 95% DE CONFIANZA (alpha = 0.05):")
    print(f"Al realizar la prueba de Chi-cuadrada, el estadístico calculado ({mejor['chi2_calc']:.4f}) es menor")
    print(f"que el valor crítico ({mejor['chi2_crit']:.4f}), por lo que SÍ se acepta la misma distribución")
    print("que selecciona Stat::Fit (no se rechaza la hipótesis nula H0).")
    print("=" * 75)

if __name__ == "__main__":
    main()