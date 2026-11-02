import numpy as np
from scipy import stats

DATOS_EJ8 = [
    0.5858, 0.8863, 0.8378, 0.3203, 0.4115, 0.2710, 0.9238, 0.1959, 0.9268, 0.6702, 0.6213, 0.4360,
    0.6279, 0.8415, 0.5786, 0.0543, 0.3567, 0.1655, 0.3880, 0.8080, 0.1931, 0.0843, 0.9152, 0.6093,
    0.7587, 0.4515, 0.3203, 0.5139, 0.7070, 0.9123, 0.1242, 0.8826, 0.9921, 0.8523, 0.6723, 0.8540,
    0.4722, 0.4781, 0.2101, 0.1680, 0.8658, 0.4028, 0.6136, 0.8720, 0.1126, 0.5857, 0.9172, 0.8943,
    0.8095, 0.6408
]

def prueba_de_series(datos, k=2, alpha=0.10):
    """
    Realiza la prueba de series bidimensional formando pares consecutivos (r_i, r_{i+1}).
    k: subintervalos por eje (total de celdas = k^2)
    """
    n_pares = len(datos) - 1
    pares = np.array([(datos[i], datos[i + 1]) for i in range(n_pares)])
    
    total_celdas = k * k
    esp = n_pares / total_celdas
    
    # Conteo de observaciones por celda
    cuadricula = np.zeros((k, k), dtype=int)
    for x, y in pares:
        ix = min(int(x * k), k - 1)
        iy = min(int(y * k), k - 1)
        cuadricula[ix, iy] += 1
        
    chi2_calc = np.sum((cuadricula - esp) ** 2 / esp)
    gl = total_celdas - 1
    chi2_crit = stats.chi2.ppf(1 - alpha, df=gl)
    pasa = chi2_calc <= chi2_crit
    
    return {
        "n_pares": n_pares,
        "k": k,
        "celdas": total_celdas,
        "esp": esp,
        "matriz_obs": cuadricula,
        "chi2_calc": chi2_calc,
        "chi2_crit": chi2_crit,
        "pasa": pasa
    }

def main():
    print("=" * 70)
    print("EJERCICIO 8: PRUEBA DE SERIES (NIVEL DE ACEPTACIÓN: 90%, alpha = 0.10)")
    print("=" * 70)
    
    res = prueba_de_series(DATOS_EJ8, k=2, alpha=0.10)
    
    print(f"Número de pares analizados: {res['n_pares']}")
    print(f"Cuadrícula: {res['k']}x{res['k']} ({res['celdas']} celdas)")
    print(f"Frecuencia esperada por celda (E_i): {res['esp']:.2f}")
    print("\nMatriz de frecuencias observadas:")
    print(res['matriz_obs'])
    print(f"\nEstadístico Chi2 calculado: {res['chi2_calc']:.4f}")
    print(f"Valor crítico Chi2 (df = {res['celdas'] - 1}, alpha = 0.10): {res['chi2_crit']:.4f}")
    
    if res['pasa']:
        print("\n-> CONCLUSIÓN: SE ACEPTA la hipótesis de independencia al 90% de confianza.")
    else:
        print("\n-> CONCLUSIÓN: SE RECHAZA la hipótesis de independencia.")
    print("=" * 70)

if __name__ == "__main__":
    main()