import math
import random
import numpy as np


def raiz_cubica(val):
    """Calcula la raíz cúbica preservando el signo para valores negativos."""
    return math.copysign(abs(val) ** (1.0 / 3.0), val)


def generar_variables(n=1000, seed=42):
    random.seed(seed)
    r = [random.random() for _ in range(n)]

    # a) x = (28*r - 1)^(1/3) - 1
    xa = [raiz_cubica(28.0 * ri - 1.0) - 1.0 for ri in r]

    # b) x = 9 + 3*sqrt(r)
    xb = [9.0 + 3.0 * math.sqrt(ri) for ri in r]

    # c) x = 8 - 3*sqrt(1 - r)
    xc = [8.0 - 3.0 * math.sqrt(1.0 - ri) for ri in r]

    # d) x = 15 - 4*sqrt(1 - r)
    xd = [15.0 - 4.0 * math.sqrt(1.0 - ri) for ri in r]

    return xa, xb, xc, xd


def main():
    print("=" * 70)
    print("EJERCICIO 16: EXPRESIONES PARA GENERAR VARIABLES ALEATORIAS")
    print("=" * 70)
    print("a) f(x) = (3/28)*(x + 1)^2    -->  x = (28*r - 1)^(1/3) - 1")
    print("b) f(x) = (2/9)*(x - 9)       -->  x = 9 + 3*sqrt(r)")
    print("c) f(x) = (2/9)*(8 - x)       -->  x = 8 - 3*sqrt(1 - r)")
    print("d) f(x) = (1/8)*(15 - x)      -->  x = 15 - 4*sqrt(1 - r)")
    print("-" * 70)

    xa, xb, xc, xd = generar_variables(n=100)

    print("Muestra de los primeros 5 valores generados para cada función:")
    print(f" - Función a: {np.round(xa[:5], 4)}")
    print(f" - Función b: {np.round(xb[:5], 4)}")
    print(f" - Función c: {np.round(xc[:5], 4)}")
    print(f" - Función d: {np.round(xd[:5], 4)}")
    print("=" * 70)


if __name__ == "__main__":
    main()