import numpy as np
from scipy import stats

# 50 números extraídos de la tabla
DATOS = [
    0.8797,
    0.9848,
    0.4557,
    0.9170,
    0.8376,
    0.3884,
    0.3469,
    0.1592,
    0.2204,
    0.6235,
    0.6289,
    0.7977,
    0.8536,
    0.5991,
    0.3681,
    0.8750,
    0.5844,
    0.8846,
    0.5461,
    0.2088,
    0.5999,
    0.8147,
    0.3410,
    0.5739,
    0.8589,
    0.9996,
    0.6431,
    0.1492,
    0.3254,
    0.1525,
    0.2006,
    0.7387,
    0.8681,
    0.0856,
    0.4720,
    0.2415,
    0.5613,
    0.5291,
    0.2258,
    0.4272,
    0.3808,
    0.0318,
    0.3188,
    0.4603,
    0.6360,
    0.9606,
    0.7401,
    0.5992,
    0.5027,
    0.0954,
]


def prueba_medias(datos, alpha=0.05):
    """Prueba de medias con distribución normal estándar N(0, 1)."""
    n = len(datos)
    media_m = float(np.mean(datos))
    mu_teorica = 0.5
    sigma_teorica = np.sqrt(1.0 / (12.0 * n))

    z_calc = (media_m - mu_teorica) / sigma_teorica
    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    lim_inf = mu_teorica - z_crit * sigma_teorica
    lim_sup = mu_teorica + z_crit * sigma_teorica
    pasa = lim_inf <= media_m <= lim_sup

    return {
        "n": n,
        "media": media_m,
        "z_calc": z_calc,
        "z_crit": z_crit,
        "lim_inf": lim_inf,
        "lim_sup": lim_sup,
        "pasa": pasa,
    }


def prueba_varianza(datos, alpha=0.05):
    """Prueba de varianza con distribución Chi-cuadrada (grados de libertad: n-1)."""
    n = len(datos)
    var_m = float(np.var(datos, ddof=1))
    gl = n - 1

    chi2_calc = (gl * var_m) / (1.0 / 12.0)
    chi2_inf = float(stats.chi2.ppf(alpha / 2, df=gl))
    chi2_sup = float(stats.chi2.ppf(1 - alpha / 2, df=gl))

    lim_inf = chi2_inf / (12.0 * gl)
    lim_sup = chi2_sup / (12.0 * gl)
    pasa = lim_inf <= var_m <= lim_sup

    return {
        "var_m": var_m,
        "chi2_calc": chi2_calc,
        "chi2_inf": chi2_inf,
        "chi2_sup": chi2_sup,
        "lim_inf": lim_inf,
        "lim_sup": lim_sup,
        "pasa": pasa,
    }


def prueba_uniformidad(datos, alpha=0.05, k=5):
    """Prueba de uniformidad con bondad de ajuste Chi-cuadrada dividiendo en k intervalos equiprobables."""
    n = len(datos)
    esp = n / k
    obs, bordes = np.histogram(datos, bins=k, range=(0.0, 1.0))

    chi2_calc = float(np.sum((obs - esp) ** 2 / esp))
    gl = k - 1
    chi2_crit = float(stats.chi2.ppf(1 - alpha, df=gl))
    pasa = chi2_calc <= chi2_crit

    return {
        "k": k,
        "bordes": bordes,
        "obs": obs,
        "esp": esp,
        "chi2_calc": chi2_calc,
        "chi2_crit": chi2_crit,
        "pasa": pasa,
    }


def main():
    print("=" * 70)
    print("EJERCICIO 7: PRUEBAS ESTADÍSTICAS A 50 NÚMEROS (CONFIANZA: 95%)")
    print("=" * 70)

    # 1. Prueba de medias
    pm = prueba_medias(DATOS)
    print(f"\n1. PRUEBA DE MEDIAS (Nivel de significancia alpha = 0.05):")
    print(f"   * Cantidad de números (n): {pm['n']}")
    print(f"   * Media muestral (x_bar): {pm['media']:.5f} (Teórica: 0.50000)")
    print(f"   * Estadístico Z_0 calculado: {pm['z_calc']:.4f}")
    print(f"   * Valor crítico Z_crit (Z_0.025): ±{pm['z_crit']:.4f}")
    print(
        f"   * Intervalo de aceptación de la media: [{pm['lim_inf']:.5f}, {pm['lim_sup']:.5f}]"
    )
    print(
        f"   -> Decisión: {'SE ACEPTA H0: La media no difiere de 0.5' if pm['pasa'] else 'SE RECHAZA H0'}"
    )

    # 2. Prueba de varianza
    pv = prueba_varianza(DATOS)
    print(f"\n2. PRUEBA DE VARIANZA (Nivel de significancia alpha = 0.05):")
    print(
        f"   * Varianza muestral (s^2): {pv['var_m']:.5f} (Teórica: {1/12:.5f})"
    )
    print(f"   * Estadístico Chi-cuadrada calculado: {pv['chi2_calc']:.4f}")
    print(
        f"   * Valores críticos de Chi-cuadrada: [{pv['chi2_inf']:.4f}, {pv['chi2_sup']:.4f}]"
    )
    print(
        f"   * Intervalo de aceptación de la varianza: [{pv['lim_inf']:.5f}, {pv['lim_sup']:.5f}]"
    )
    print(
        f"   -> Decisión: {'SE ACEPTA H0: La varianza no difiere de 1/12' if pv['pasa'] else 'SE RECHAZA H0'}"
    )

    # 3. Prueba de uniformidad
    pu = prueba_uniformidad(DATOS, k=5)
    print(
        f"\n3. PRUEBA DE UNIFORMIDAD CHI-CUADRADA (k = 5 intervalos, alpha = 0.05):"
    )
    print(f"   * Frecuencia esperada por intervalo (E_i): {pu['esp']:.1f}")
    for i in range(pu["k"]):
        rango = f"[{pu['bordes'][i]:.1f} - {pu['bordes'][i+1]:.1f})"
        print(f"     - Intervalo {i+1} {rango:15}: O_{i+1} = {pu['obs'][i]}")
    print(f"   * Estadístico Chi-cuadrada calculado: {pu['chi2_calc']:.4f}")
    print(f"   * Valor crítico Chi-cuadrada (df = 4, alpha = 0.05): {pu['chi2_crit']:.4f}")
    print(
        f"   -> Decisión: {'SE ACEPTA H0: Los números se distribuyen uniformemente' if pu['pasa'] else 'SE RECHAZA H0'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()