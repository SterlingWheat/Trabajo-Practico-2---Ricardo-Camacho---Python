import math
import random
import numpy as np


def inversa_tabla(r, valores, probs):
    """Método de la transformada inversa discreta general por acumulación."""
    acumulada = np.cumsum(probs)
    for v, f_x in zip(valores, acumulada):
        if r <= f_x:
            return v
    return valores[-1]


def generar_100_muestras(seed=42):
    random.seed(seed)
    N = 100

    # Generamos los 100 números pseudoaleatorios base
    r = [random.random() for _ in range(N)]

    # a) p(x) = (1/5)*(4/5)^(x-1), x = 1, 2, 3...
    res_a = [math.ceil(math.log(1 - ri) / math.log(0.8)) for ri in r]

    # b) p(x) = (1/2)*(1/2)^x, x = 0, 1, 2...
    res_b = [math.floor(math.log(1 - ri) / math.log(0.5)) for ri in r]

    # c) p(x) = (20 + 10x)/200, x = 0, 1, 2, 3, 4
    vals_c = [0, 1, 2, 3, 4]
    probs_c = [(20 + 10 * x) / 200 for x in vals_c]
    res_c = [inversa_tabla(ri, vals_c, probs_c) for ri in r]

    # d) p(x) = (18 - 4x)/50, x = 0, 1, 2, 3, 4
    vals_d = [0, 1, 2, 3, 4]
    probs_d = [(18 - 4 * x) / 50 for x in vals_d]
    res_d = [inversa_tabla(ri, vals_d, probs_d) for ri in r]

    # e) p(x) = (3x + 8)/100, x = 2, 3, 4, 5, 6
    vals_e = [2, 3, 4, 5, 6]
    probs_e = [(3 * x + 8) / 100 for x in vals_e]
    res_e = [inversa_tabla(ri, vals_e, probs_e) for ri in r]

    # f) p(x) = 1/6, x = 6, 7, 8, 9, 10, 11
    res_f = [6 + math.floor(6 * ri) for ri in r]

    # g) Geométrica truncada con p = 0.3 y n = 10
    p_param = 0.3
    n_param = 10
    q = 1.0 - p_param
    den = 1.0 - (q**n_param)
    res_g = [
        math.ceil(math.log(1.0 - ri * den) / math.log(q)) for ri in r
    ]

    return {
        "a": res_a,
        "b": res_b,
        "c": res_c,
        "d": res_d,
        "e": res_e,
        "f": res_f,
        "g": res_g,
    }


def main():
    muestras = generar_100_muestras()

    print("=" * 80)
    print("EJERCICIO 14: MÉTODO DE LA TRANSFORMADA INVERSA DISCRETA (100 NÚMEROS)")
    print("=" * 80)

    for inciso, datos in muestras.items():
        valores_unicos, frecuencias = np.unique(datos, return_counts=True)
        print(f"\nInciso ({inciso}):")
        print(f"  * Primeros 15 valores generados: {datos[:15]}")
        print(f"  * Media muestral: {np.mean(datos):.4f}")
        print(f"  * Conteo de frecuencias observadas (100 datos):")
        distribucion = dict(zip(valores_unicos, frecuencias))
        print(f"    {distribucion}")

    print("=" * 80)


if __name__ == "__main__":
    main()