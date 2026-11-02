def periodo_orden_1(siguiente_fn, x0):
    """Calcula el periodo de un generador congruencial de primer orden."""
    historial = {}
    actual = x0
    paso = 0
    historial[actual] = paso
    secuencia = [actual]

    while True:
        paso += 1
        siguiente = siguiente_fn(actual)
        secuencia.append(siguiente)

        if siguiente in historial:
            inicio_ciclo = historial[siguiente]
            longitud_periodo = paso - inicio_ciclo
            return longitud_periodo, secuencia, inicio_ciclo

        historial[siguiente] = paso
        actual = siguiente


def periodo_orden_2(siguiente_fn, x0, x1):
    """Calcula el periodo de un generador congruencial de segundo orden (pares de estados)."""
    historial = {}
    previo, actual = x0, x1
    paso = 1
    historial[(previo, actual)] = paso
    secuencia = [previo, actual]

    while True:
        paso += 1
        siguiente = siguiente_fn(actual, previo)
        secuencia.append(siguiente)

        par_estado = (actual, siguiente)
        if par_estado in historial:
            inicio_ciclo = historial[par_estado]
            longitud_periodo = paso - inicio_ciclo
            return longitud_periodo, secuencia, inicio_ciclo

        historial[par_estado] = paso
        previo, actual = actual, siguiente


def main():
    print("=" * 65)
    print("DETERMINACIÓN DEL CICLO O PERIODO DE VIDA DE GENERADORES")
    print("=" * 65)

    # a) x_{i+1} = (21 * x_i + 15) mod 31 con x_0 = 21
    periodo_a, sec_a, ini_a = periodo_orden_1(lambda x: (21 * x + 15) % 31, 21)
    print(f"\na) x_{{i+1}} = (21*x_i + 15) mod 31 | x_0 = 21")
    print(f"   -> Periodo: {periodo_a}")
    print(f"   -> Primeros valores: {sec_a[:12]}...")

    # b) x_{i+1} = (13 * x_i + 9) mod 128 con x_0 = 7
    periodo_b, sec_b, ini_b = periodo_orden_1(lambda x: (13 * x + 9) % 128, 7)
    print(f"\nb) x_{{i+1}} = (13*x_i + 9) mod 128 | x_0 = 7")
    print(f"   -> Periodo: {periodo_b}")
    print(f"   -> Primeros valores: {sec_b[:12]}...")

    # c) x_{i+1} = (17 * x_i) mod 31 con x_0 = 23
    periodo_c, sec_c, ini_c = periodo_orden_1(lambda x: (17 * x) % 31, 23)
    print(f"\nc) x_{{i+1}} = (17*x_i) mod 31 | x_0 = 23")
    print(f"   -> Periodo: {periodo_c}")
    print(f"   -> Primeros valores: {sec_c[:12]}...")

    # d) x_{i+1} = (121 + x_i) mod 256 con x_0 = 17
    periodo_d, sec_d, ini_d = periodo_orden_1(lambda x: (121 + x) % 256, 17)
    print(f"\nd) x_{{i+1}} = (121 + x_i) mod 256 | x_0 = 17")
    print(f"   -> Periodo: {periodo_d}")
    print(f"   -> Primeros valores: {sec_d[:12]}...")

    # e) x_{i+1} = (21 * x_i + 15 * x_{i-1}) mod 64 con x_0 = 21, x_1 = 43
    periodo_e, sec_e, ini_e = periodo_orden_2(
        lambda xi, xi_prev: (21 * xi + 15 * xi_prev) % 64, 21, 43
    )
    print(
        f"\ne) x_{{i+1}} = (21*x_i + 15*x_{{i-1}}) mod 64 | x_0 = 21, x_1 = 43"
    )
    print(f"   -> Periodo: {periodo_e}")
    print(f"   -> Primeros valores: {sec_e[:12]}...")
    print("=" * 65)


if __name__ == "__main__":
    main()