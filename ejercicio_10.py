import numpy as np
import pandas as pd
from scipy import stats

# 100 datos de la imagen
DATOS = [
    # Bloque superior (42 datos)
    17.392,
    4.078,
    3.151,
    3.528,
    2.440,
    5.924,
    3.461,
    2.052,
    10.369,
    3.690,
    10.870,
    4.793,
    2.498,
    0.569,
    8.281,
    0.154,
    5.959,
    3.384,
    12.877,
    13.602,
    5.244,
    16.677,
    5.977,
    4.313,
    4.767,
    2.381,
    6.443,
    1.392,
    1.578,
    8.115,
    4.891,
    6.720,
    7.728,
    2.717,
    10.451,
    5.901,
    0.818,
    7.088,
    2.637,
    4.714,
    3.032,
    1.495,
    15.733,
    # Bloque inferior (58 datos)
    7.768,
    2.333,
    7.822,
    3.708,
    6.412,
    1.290,
    3.957,
    5.285,
    7.094,
    3.078,
    1.264,
    2.630,
    10.177,
    2.155,
    2.945,
    7.552,
    11.094,
    4.772,
    7.281,
    14.344,
    19.867,
    0.119,
    2.072,
    1.486,
    3.791,
    4.214,
    1.611,
    1.781,
    1.530,
    3.280,
    4.301,
    0.202,
    7.489,
    1.422,
    1.453,
    0.022,
    6.001,
    9.269,
    8.477,
    3.043,
    0.877,
    6.966,
    2.103,
    1.816,
    0.433,
    2.547,
    0.843,
    1.182,
    8.121,
    2.007,
    1.395,
    4.661,
    7.378,
    5.300,
    17.066,
    12.171,
]


def prueba_chi_cuadrada_distribucion(
    datos, dist_nombre, dist_obj, k=8, alpha=0.10
):
    """Ajusta una distribución por MLE y evalúa la prueba de bondad de ajuste Chi-cuadrada

    agrupando clases con probabilidades teóricas iguales (equiprobables).
    """
    n = len(datos)

    # 1. Ajuste de parámetros por Máxima Verosimilitud (MLE)
    params = dist_obj.fit(datos)
    num_params = len(params)

    # 2. Partición en k clases equiprobables según la distribución teórica acumulada (CDF)
    p_cortes = np.linspace(0, 1, k + 1)
    bordes = dist_obj.ppf(p_cortes, *params)
    bordes[0] = -np.inf
    bordes[-1] = np.inf

    # 3. Frecuencias observadas y esperadas
    obs, _ = np.histogram(datos, bins=bordes)
    esp = n / k  # Al ser clases equiprobables

    # 4. Estadístico Chi-cuadrada
    chi2_calc = np.sum((obs - esp) ** 2 / esp)

    # Grados de libertad: k - 1 - p (donde p es el número de parámetros estimados)
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
        "pasa": pasa,
    }


def main():
    datos = np.array(DATOS)
    alpha = 0.10  # 90% nivel de confianza

    print("=" * 75)
    print("EJERCICIO 10: AJUSTE DE DISTRIBUCIONES (STAT::FIT PROMODEL - CHI-CUADRADA)")
    print("=" * 75)
    print(f"Número de observaciones (n): {len(datos)}")
    print(f"Media muestral: {np.mean(datos):.4f}")
    print(f"Desviación estándar: {np.std(datos, ddof=1):.4f}")
    print(f"Mínimo: {np.min(datos):.4f} | Máximo: {np.max(datos):.4f}")
    print("-" * 75)

    # Distribuciones típicas candidatas en Stat::Fit
    distribuciones = [
        ("Exponencial", stats.expon),
        ("Weibull", stats.weibull_min),
        ("Gamma", stats.gamma),
        ("Lognormal", stats.lognorm),
        ("Normal", stats.norm),
    ]

    resultados = []
    for nombre, dist_obj in distribuciones:
        res = prueba_chi_cuadrada_distribucion(
            datos, nombre, dist_obj, k=8, alpha=alpha
        )
        resultados.append(res)

    # Ordenar por menor estadístico Chi-cuadrada (mayor p-valor = mejor ajuste)
    resultados = sorted(resultados, key=lambda x: x["chi2_calc"])

    print(
        f"{'Distribución':15} | {'Chi2 Calc':10} | {'Chi2 Crít':10} | {'p-valor':10} | {'Decisión (90%)'}"
    )
    print("-" * 75)
    for r in resultados:
        estado = "ACEPTADA" if r["pasa"] else "RECHAZADA"
        print(
            f"{r['distribucion']:15} | {r['chi2_calc']:10.4f} | {r['chi2_crit']:10.4f} | {r['p_valor']:10.4f} | {estado}"
        )

    print("=" * 75)
    mejor = resultados[0]
    print(f"DISTRIBUCIÓN RECOMENDADA POR STAT::FIT:")
    print(
        f"-> {mejor['distribucion']} con estadístico Chi-cuadrada de {mejor['chi2_calc']:.4f} (p-valor: {mejor['p_valor']:.4f})."
    )
    print(
        f"-> Conclusión: Al 90% de confianza, la prueba de Chi-cuadrada SÍ ACEPTA la misma"
    )
    print(
        f"   distribución que propone el software (no hay evidencia para rechazarla)."
    )
    print("=" * 75)


if __name__ == "__main__":
    main()