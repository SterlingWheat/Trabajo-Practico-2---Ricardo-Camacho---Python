import math
import random
import numpy as np


def generar_variables(n=1000, seed=42):
    random.seed(seed)
    
    # Lista de números pseudoaleatorios r ~ U(0, 1)
    r = [random.random() for _ in range(n)]
    
    # Expresiones matemáticas deducidas
    # a) x = sqrt(-10 * ln(1 - r))
    xa = [math.sqrt(-10.0 * math.log(1.0 - ri)) for ri in r]
    
    # b) x = (-ln(1 - r) / 12)^(1/3)
    xb = [(-math.log(1.0 - ri) / 12.0) ** (1.0 / 3.0) for ri in r]
    
    # c) x = (-ln(1 - r))^(1/4)
    xc = [(-math.log(1.0 - ri)) ** (1.0 / 4.0) for ri in r]
    
    return xa, xb, xc


def main():
    print("=" * 70)
    print("EJERCICIO 15: EXPRESIONES MATEMÁTICAS POR TRANSFORMADA INVERSA")
    print("=" * 70)
    print("a) f(x) = (x/5) * e^(-x^2 / 10)  -->  x = sqrt(-10 * ln(1 - r))")
    print("b) f(x) = 36x^2 * e^(-12x^3)    -->  x = (-ln(1 - r) / 12)^(1/3)")
    print("c) f(x) = 4x^3 * e^(-x^4)       -->  x = (-ln(1 - r))^(1/4)")
    print("-" * 70)
    
    xa, xb, xc = generar_variables(n=100)
    
    print("Muestra de los primeros 5 valores generados para cada función:")
    print(f" - Función a: {np.round(xa[:5], 4)}")
    print(f" - Función b: {np.round(xb[:5], 4)}")
    print(f" - Función c: {np.round(xc[:5], 4)}")
    print("=" * 70)


if __name__ == "__main__":
    main()