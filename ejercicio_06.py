import numpy as np
from scipy import stats


def generar_serie(a=121, c=553, m=177, x0=23):
    """
    Genera los valores enteros x_i y los números pseudoaleatorios r_i en [0, 1)
    hasta detectar la repetición del ciclo o periodo de vida.
    """
    historial = {}
    x_actual = x0
    secuencia_x = [x_actual]
    paso = 0
    historial[x_actual] = paso

    while True:
        paso += 1
        x_siguiente = (a * x_actual + c) % m

        if x_siguiente in historial:
            inicio_ciclo = historial[x_siguiente]
            periodo = paso - inicio_ciclo
            break

        historial[x_siguiente] = paso
        secuencia_x.append(x_siguiente)
        x_actual = x_siguiente

    # Convertir los valores generados del periodo a números pseudoaleatorios r_i = x_i / m
    valores_periodo = secuencia_x[inicio_ciclo:]
    r_periodo = np.array(valores_periodo) / m

    return periodo, secuencia_x, r_periodo


def prueba_medias(r, alpha=0.05):
    """Prueba de medias con distribución normal estándar."""
    n = len(r)
    media_muestral = np.mean(r)
    media_teorica = 0.5
    desv_teorica = np.sqrt(1.0 / (12.0 * n))

    z_calculado = (media_muestral - media_teorica) / desv_teorica
    z_critico = stats.norm.ppf(1 - alpha / 2)

    pasa = abs(z_calculado) <= z_critico
    lim_inf = media_teorica - z_critico * desv_teorica
    lim_sup = media_teorica + z_critico * desv_teorica

    return {
        "media": media_muestral,
        "Z_calc": z_calculado,
        "Z_crit": z_critico,
        "lim_inf": lim_inf,
        "lim_sup": lim_sup,
        "pasa": pasa,
    }


def prueba_varianza(r, alpha=0.05):
    """Prueba de varianza con distribución Chi-cuadrada."""
    n = len(r)
    varianza_muestral = np.var(r, ddof=1)
    grados_libertad = n - 1

    chi2_calculado = (grados_libertad * varianza_muestral) / (1.0 / 12.0)
    chi2_inf = stats.chi2.ppf(alpha / 2, df=grados_libertad)
    chi2_sup = stats.chi2.ppf(1 - alpha / 2, df=grados_libertad)

    lim_inf = chi2_inf / (12.0 * grados_libertad)
    lim_sup = chi2_sup / (12.0 * grados_libertad)

    pasa = chi2_inf <= chi2_calculado <= chi2_sup

    return {
        "varianza": varianza_muestral,
        "chi2_calc": chi2_calculado,
        "chi2_inf": chi2_inf,
        "chi2_sup": chi2_sup,
        "lim_inf": lim_inf,
        "lim_sup": lim_sup,
        "pasa": pasa,
    }


def prueba_uniformidad_chi2(r, alpha=0.05, num_intervalos=None):
    """Prueba de uniformidad mediante Chi-cuadrada con k intervalos equiprobables."""
    n = len(r)
    if num_intervalos is None:
        num_intervalos = max(5, int(np.sqrt(n)))

    # Frecuencia esperada por intervalo
    esperada = n / num_intervalos
    frecuencias_obs, _ = np.histogram(r, bins=num_intervalos, range=(0.0, 1.0))

    chi2_calculado = np.sum((frecuencias_obs - esperada) ** 2 / esperada)
    grados_libertad = num_intervalos - 1
    chi2_critico = stats.chi2.ppf(1 - alpha, df=grados_libertad)

    pasa = chi2_calculado <= chi2_critico

    return {
        "k": num_intervalos,
        "obs": frecuencias_obs,
        "esp": esperada,
        "chi2_calc": chi2_calculado,
        "chi2_crit": chi2_critico,
        "pasa": pasa,
    }


def main():
    print("=" * 65)
    print("EJERCICIO 6: SERIE CONGRUENCIAL x_{i+1} = (553 + 121*x_i) mod 177")
    print("=" * 65)

    # a) Determinar el ciclo o periodo de vida
    periodo, secuencia, r = generar_serie(a=121, c=553, m=177, x0=23)
    print(f"\na) CICLO O PERIODO DE VIDA:")
    print(f"   * Longitud del periodo (N): {periodo}")
    print(f"   * Primeros 10 valores de x_i: {secuencia[:10]}")
    print(f"   * Primeros 5 números r_i generados: {np.round(r[:5], 4)}")

    # b) Pruebas estadísticas (Nivel de significancia 5%, alpha = 0.05)
    print("\n" + "-" * 65)
    print("b) PRUEBAS ESTADÍSTICAS (alpha = 0.05):")
    print("-" * 65)

    # 1. Prueba de Medias
    res_m = prueba_medias(r)
    print(f"1. PRUEBA DE MEDIAS:")
    print(f"   - Media de la muestra: {res_m['media']:.5f}")
    print(f"   - Estadístico Z calculado: {res_m['Z_calc']:.4f}")
    print(
        f"   - Límites de aceptación: [{res_m['lim_inf']:.4f}, {res_m['lim_sup']:.4f}]"
    )
    print(
        f"   - Decisión: {'ACEPTADA (La media no es significativamente diferente de 0.5)' if res_m['pasa'] else 'RECHAZADA'}"
    )

    # 2. Prueba de Varianza
    res_v = prueba_varianza(r)
    print(f"\n2. PRUEBA DE VARIANZA:")
    print(
        f"   - Varianza de la muestra: {res_v['varianza']:.5f} (Teórica: {1/12:.5f})"
    )
    print(f"   - Estadístico Chi2 calculado: {res_v['chi2_calc']:.4f}")
    print(
        f"   - Intervalo crítico Chi2: [{res_v['chi2_inf']:.4f}, {res_v['chi2_sup']:.4f}]"
    )
    print(
        f"   - Decisión: {'ACEPTADA (La varianza es consistente con 1/12)' if res_v['pasa'] else 'RECHAZADA'}"
    )

    # 3. Prueba de Uniformidad
    res_u = prueba_uniformidad_chi2(r)
    print(f"\n3. PRUEBA DE UNIFORMIDAD (Chi-Cuadrada):")
    print(
        f"   - Intervalos (k): {res_u['k']} (Frecuencia esperada: {res_u['esp']:.2f})"
    )
    print(f"   - Frecuencias observadas por intervalo: {res_u['obs']}")
    print(
        f"   - Estadístico Chi2: {res_u['chi2_calc']:.4f} <= Crítico: {res_u['chi2_crit']:.4f}"
    )
    print(
        f"   - Decisión: {'ACEPTADA (Los datos provienen de una distribución uniforme U(0,1))' if res_u['pasa'] else 'RECHAZADA'}"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()