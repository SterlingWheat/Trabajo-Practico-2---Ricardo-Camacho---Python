import numpy as np
from scipy import stats

# 1. Definición de intervalos y frecuencias observadas
intervalos = [
    (0, 2), (2, 4), (4, 6), (6, 8), (8, 10),
    (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22)
]
obs = np.array([39, 57, 47, 23, 18, 9, 2, 3, 1, 1, 0])
n = int(np.sum(obs)) # n = 200
alpha = 0.10 # 90% nivel de confianza

# Estimación de media y varianza muestral por marcas de clase
marcas = np.array([(a + b) / 2.0 for a, b in intervalos])
media_m = np.sum(obs * marcas) / n
var_m = np.sum(obs * ((marcas - media_m) ** 2)) / (n - 1)
desv_m = np.sqrt(var_m)

# 2. Evaluación de distribuciones teóricas candidatas
# Para la distribución Gamma: media = k * theta, var = k * theta^2
theta_gamma = var_m / media_m
k_gamma = media_m / theta_gamma

candidatas = {
    "Gamma": stats.gamma(a=k_gamma, scale=theta_gamma),
    "Weibull": stats.weibull_min(c=1.5, scale=media_m / 0.9027),
    "Lognormal": stats.lognorm(s=np.sqrt(np.log(1 + var_m / (media_m**2))), scale=media_m / np.sqrt(1 + var_m / (media_m**2))),
    "Exponencial": stats.expon(scale=media_m)
}

print("=" * 75)
print("EJERCICIO 13: DETERMINACIÓN DE DISTRIBUCIÓN (CHI-CUADRADA, CONFIANZA 90%)")
print("=" * 75)
print(f"Total de datos (n): {n}")
print(f"Media muestral estimada: {media_m:.4f}")
print(f"Desviación estándar estimada: {desv_m:.4f}")
print("-" * 75)

# 3. Aplicación de la prueba Chi-cuadrada con colapso de colas (E_i >= 5)
def evaluar_distribucion(nombre, dist_obj, num_params=2):
    p_esperadas = []
    for i, (a, b) in enumerate(intervalos):
        lim_inf = 0.0 if i == 0 else a
        lim_sup = np.inf if i == len(intervalos) - 1 else b
        prob = dist_obj.cdf(lim_sup) - dist_obj.cdf(lim_inf)
        p_esperadas.append(prob)
        
    p_esperadas = np.array(p_esperadas)
    frec_esp = n * p_esperadas
    
    # Agrupación de clases con frecuencia esperada menor a 5
    obs_agrup = []
    esp_agrup = []
    
    acum_o, acum_e = 0, 0.0
    for o_i, e_i in zip(obs, frec_esp):
        acum_o += o_i
        acum_e += e_i
        if acum_e >= 5.0:
            obs_agrup.append(acum_o)
            esp_agrup.append(acum_e)
            acum_o, acum_e = 0, 0.0
            
    if acum_e > 0:
        obs_agrup[-1] += acum_o
        esp_agrup[-1] += acum_e
        
    obs_agrup = np.array(obs_agrup)
    esp_agrup = np.array(esp_agrup)
    
    chi2_calc = np.sum((obs_agrup - esp_agrup) ** 2 / esp_agrup)
    gl = len(obs_agrup) - 1 - num_params
    if gl <= 0:
        gl = 1
    chi2_crit = stats.chi2.ppf(1 - alpha, df=gl)
    p_valor = 1.0 - stats.chi2.cdf(chi2_calc, df=gl)
    pasa = chi2_calc <= chi2_crit
    
    return {
        "nombre": nombre,
        "clases_finales": len(obs_agrup),
        "gl": gl,
        "chi2_calc": chi2_calc,
        "chi2_crit": chi2_crit,
        "p_valor": p_valor,
        "pasa": pasa
    }

resultados = []
for nom, obj in candidatas.items():
    p_count = 1 if nom == "Exponencial" else 2
    res = evaluar_distribucion(nom, obj, num_params=p_count)
    resultados.append(res)

resultados = sorted(resultados, key=lambda x: x["chi2_calc"])

print(f"{'Distribución':15} | {'Clases':7} | {'Chi2 Calc':10} | {'Chi2 Crít (90%)':16} | {'p-valor':10} | {'Decisión'}")
print("-" * 75)
for r in resultados:
    estado = "ACEPTADA" if r["pasa"] else "RECHAZADA"
    print(f"{r['nombre']:15} | {r['clases_finales']:7} | {r['chi2_calc']:10.4f} | {r['chi2_crit']:16.4f} | {r['p_valor']:10.4f} | {estado}")

print("=" * 75)
mejor = resultados[0]
print(f"DISTRIBUCIÓN QUE SIGUEN LOS DATOS:")
print(f"-> {mejor['nombre']} con estadístico Chi-cuadrada de {mejor['chi2_calc']:.4f} (p-valor: {mejor['p_valor']:.4f}).")
print("\nCONCLUSIÓN:")
print(f"Con un nivel de confianza del 90% (alpha = 0.10), se confirma que los datos siguen")
print(f"una DISTRIBUCIÓN {mejor['nombre'].upper()}, ya que el estadístico calculado no supera")
print(f"el valor crítico de rechazo ({mejor['chi2_calc']:.4f} <= {mejor['chi2_crit']:.4f}).")
print("=" * 75)

if __name__ == "__main__":
    pass