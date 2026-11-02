import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simular_demanda(r):
    """Método de la transformada inversa para la demanda discreta."""
    if r < 0.10:
        return 0
    elif r < 0.25:
        return 1
    elif r < 0.50:
        return 2
    elif r < 0.70:
        return 3
    elif r < 0.85:
        return 4
    elif r < 0.93:
        return 5
    else:
        return 6

def main():
    horas = 100
    num_corridas = 5
    precio = 6.0
    costo = 3.5
    margen = precio - costo # $2.5 por unidad
    alpha = 0.05            # Nivel de confianza del 95%
    
    # Semillas reproducibles para cada corrida
    semillas = [101, 202, 303, 404, 505]
    
    t = np.arange(1, horas + 1)
    promedios_corridas = []
    medias_finales = []
    intervalos_individuales = []
    
    plt.figure(figsize=(13, 7))
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    print("=" * 80)
    print("EJERCICIO 17: SIMULACIÓN DE UTILIDAD PROMEDIO POR HORA (5 RÉPLICAS)")
    print("=" * 80)
    
    for i in range(num_corridas):
        np.random.seed(semillas[i])
        r = np.random.uniform(0, 1, size=horas)
        demandas = np.array([simular_demanda(val) for val in r])
        utilidades = demandas * margen
        
        # Promedio acumulado (estabilización)
        prom_acumulado = np.cumsum(utilidades) / t
        promedios_corridas.append(prom_acumulado)
        
        media_corrida = float(np.mean(utilidades))
        s_corrida = float(np.std(utilidades, ddof=1))
        medias_finales.append(media_corrida)
        
        # Intervalo de confianza al 95% para la corrida i (n = 100 observaciones)
        t_crit = stats.t.ppf(1 - alpha / 2, df=horas - 1)
        error_std = s_corrida / np.sqrt(horas)
        ic_inf = media_corrida - t_crit * error_std
        ic_sup = media_corrida + t_crit * error_std
        intervalos_individuales.append((ic_inf, ic_sup))
        
        # Graficar trayectoria acumulada
        plt.plot(t, prom_acumulado, label=f"Corrida {i+1} (Final: ${media_corrida:.2f})", color=colores[i], linewidth=1.8)
        
        print(f"Corrida {i+1}:")
        print(f"  * Utilidad media final (hora 100): ${media_corrida:.3f}")
        print(f"  * Desviación estándar (s): ${s_corrida:.3f}")
        print(f"  * Intervalo de Confianza al 95%: [${ic_inf:.3f}, ${ic_sup:.3f}]")
    
    # Línea del valor teórico esperado
    plt.axhline(y=6.675, color='black', linestyle='--', linewidth=2, label="Valor Teórico Esperado ($6.675)")
    
    # Intervalo de confianza global para las 5 réplicas
    medias_finales = np.array(medias_finales)
    media_global = float(np.mean(medias_finales))
    s_global = float(np.std(medias_finales, ddof=1))
    t_crit_global = stats.t.ppf(1 - alpha / 2, df=num_corridas - 1)
    error_global = s_global / np.sqrt(num_corridas)
    ic_glob_inf = media_global - t_crit_global * error_global
    ic_glob_sup = media_global + t_crit_global * error_global
    
    # Sombreado del intervalo global en la gráfica
    plt.axhspan(ic_glob_inf, ic_glob_sup, color='gray', alpha=0.25, label=f"IC Global 95% [${ic_glob_inf:.2f}, ${ic_glob_sup:.2f}]")
    
    plt.title("Gráfica de Estabilización de la Utilidad Promedio por Hora (100 Horas)", fontsize=14)
    plt.xlabel("Horas de Trabajo Simuladas", fontsize=12)
    plt.ylabel("Utilidad Promedio Acumulada ($/hora)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    
    plt.savefig("estabilizacion_utilidad_ejercicio_17.png", dpi=300)
    print("\nGráfica guardada como 'estabilizacion_utilidad_ejercicio_17.png'.")
    plt.show()
    
    print("-" * 80)
    print("ANÁLISIS ESTADÍSTICO GLOBAL (ENTRE RÉPLICAS):")
    print(f"  * Media de las 5 corridas: ${media_global:.3f}")
    print(f"  * Desviación estándar entre corridas (S_xbar): ${s_global:.3f}")
    print(f"  * Intervalo de Confianza Global (95%): [${ic_glob_inf:.3f}, ${ic_glob_sup:.3f}]")
    print("=" * 80)

if __name__ == "__main__":
    main()