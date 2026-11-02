import random
import numpy as np
from scipy import stats

def ejecutar_bateria_pruebas(datos, alpha=0.05):
    n = len(datos)
    resultados = {}
    
    # 1. Prueba de Medias
    media = float(np.mean(datos))
    sigma_teorica = np.sqrt(1.0 / (12.0 * n))
    z_calc = (media - 0.5) / sigma_teorica
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    pasa_media = abs(z_calc) <= z_crit
    resultados['media'] = {
        'val': media, 'z_calc': z_calc, 'z_crit': z_crit, 'pasa': pasa_media
    }
    
    # 2. Prueba de Varianza
    var = float(np.var(datos, ddof=1))
    gl_v = n - 1
    chi2_v_calc = (gl_v * var) / (1.0 / 12.0)
    chi2_v_inf = float(stats.chi2.ppf(alpha / 2, df=gl_v))
    chi2_v_sup = float(stats.chi2.ppf(1 - alpha / 2, df=gl_v))
    pasa_var = chi2_v_inf <= chi2_v_calc <= chi2_v_sup
    resultados['varianza'] = {
        'val': var, 'chi2_calc': chi2_v_calc, 'lims': (chi2_v_inf, chi2_v_sup), 'pasa': pasa_var
    }
    
    # 3. Prueba de Uniformidad (k = 10 intervalos)
    k = 10
    esp = n / k
    obs, _ = np.histogram(datos, bins=k, range=(0.0, 1.0))
    chi2_u_calc = float(np.sum((obs - esp) ** 2 / esp))
    chi2_u_crit = float(stats.chi2.ppf(1 - alpha, df=k - 1))
    pasa_unif = chi2_u_calc <= chi2_u_crit
    resultados['uniformidad'] = {
        'k': k, 'obs': obs, 'chi2_calc': chi2_u_calc, 'chi2_crit': chi2_u_crit, 'pasa': pasa_unif
    }
    
    # 4. Prueba de Independencia (Prueba de Rachas / Runs test sobre la media 0.5)
    secuencia_binaria = [1 if x >= 0.5 else 0 for x in datos]
    n1 = sum(secuencia_binaria)
    n2 = n - n1
    rachas = 1 + sum(1 for i in range(1, n) if secuencia_binaria[i] != secuencia_binaria[i - 1])
    
    mu_rachas = (2.0 * n1 * n2) / n + 1.0
    var_rachas = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n**2 * (n - 1.0))
    z_rachas = (rachas - mu_rachas) / np.sqrt(var_rachas)
    pasa_indep = abs(z_rachas) <= z_crit
    resultados['independencia'] = {
        'rachas': rachas, 'mu_r': mu_rachas, 'z_calc': z_rachas, 'z_crit': z_crit, 'pasa': pasa_indep
    }
    
    return resultados

def main():
    # Semilla fija para reproducibilidad
    random.seed(42)
    numeros = [random.random() for _ in range(100)]
    
    print("=" * 70)
    print("EJERCICIO 9: ANÁLISIS DE 100 NÚMEROS PSEUDOALEATORIOS (PYTHON)")
    print("=" * 70)
    print("Primeros 10 números generados:")
    print(np.round(numeros[:10], 4))
    
    res = ejecutar_bateria_pruebas(numeros, alpha=0.05)
    
    print("\n1. PRUEBA DE MEDIAS:")
    print(f"   Media: {res['media']['val']:.4f} | Z_calc: {res['media']['z_calc']:.4f} (Crítico: ±{res['media']['z_crit']:.4f})")
    print(f"   -> {'APROBADA' if res['media']['pasa'] else 'RECHAZADA'}")
    
    print("\n2. PRUEBA DE VARIANZA:")
    print(f"   Varianza: {res['varianza']['val']:.4f} | Chi2_calc: {res['varianza']['chi2_calc']:.4f}")
    print(f"   Límites Chi2: [{res['varianza']['lims'][0]:.2f}, {res['varianza']['lims'][1]:.2f}]")
    print(f"   -> {'APROBADA' if res['varianza']['pasa'] else 'RECHAZADA'}")
    
    print("\n3. PRUEBA DE UNIFORMIDAD:")
    print(f"   Frecuencias observadas (10 clases): {res['uniformidad']['obs']}")
    print(f"   Chi2_calc: {res['uniformidad']['chi2_calc']:.4f} <= Crítico: {res['uniformidad']['chi2_crit']:.4f}")
    print(f"   -> {'APROBADA' if res['uniformidad']['pasa'] else 'RECHAZADA'}")
    
    print("\n4. PRUEBA DE INDEPENDENCIA (Rachas):")
    print(f"   Total de rachas observadas: {res['independencia']['rachas']} (Esperadas: {res['independencia']['mu_r']:.2f})")
    print(f"   Z_calc: {res['independencia']['z_calc']:.4f} (Crítico: ±{res['independencia']['z_crit']:.4f})")
    print(f"   -> {'APROBADA' if res['independencia']['pasa'] else 'RECHAZADA'}")
    
    print("\n" + "=" * 70)
    todas_pasan = all(v['pasa'] for v in res.values())
    if todas_pasan:
        print("RESPUESTA A LA PREGUNTA:")
        print("Sí, bajo este análisis estadístico se considera que el generador es")
        print("completamente CONFIABLE. Los números no muestran evidencia estadística de")
        print("sesgo en su valor medio, dispersión anormal, falta de uniformidad ni dependencia.")
    print("=" * 70)

if __name__ == "__main__":
    main()