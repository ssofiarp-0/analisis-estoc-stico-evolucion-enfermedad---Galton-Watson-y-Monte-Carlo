# Este programa realiza una SIMULACIÓN MONTE CARLO para estudiar
# el comportamiento estocástico de una enfermedad y su dinámica de absorción a largo plazo.

# PARÁMETROS DEL MODELO BASE:
# n0: tamaño inicial de la población.
# pI: proporción inicial de enfermos respecto a la población inicial (n0).
# lambda_descendientes: parámetro lambda de una distribución Poisson que rige la descendencia de cada progenitor.
# p0: probabilidad de que el sucesor desarrolle la enfermedad si el progenitor está sano (mutación de novo/transmisión).
# p1: probabilidad de que el sucesor desarrolle la enfermedad si el progenitor está enfermo (herencia).
# m0: proporción de mortalidad prematura basal de un individuo sano (causas ajenas a la enfermedad).
# m1: proporción de mortalidad de un individuo enfermo (sin tratamiento/fármaco).

# CONSIDERACIONES DEL PROCESO ESTOCÁSTICO DE GALTON-WATSON:
#   1. La descendencia por progenitor no es constante; sigue una distribución estocástica Poisson(lambda).
#   2. Supervivencia condicional: no todos los individuos enfermos mueren (m1), ni todos los sanos sobreviven (m0).
#   3. La dinámica posee un estado absorbente persistente en N = 0 (extinción total).

# ¿POR QUÉ REALIZO UNA SIMULACIÓN MONTE CARLO?
# Una única simulación representa solamente un escenario posible condicionado por una semilla aleatoria.
# Al ejecutar M = 500 simulaciones compuestas por trayectorias independientes:
#   1. Eliminamo el sesgo de casos aislados ("outliers" o fluctuaciones fortuitas).
#   2. Calculo la esperanza matemática E[N_t] y la tendencia media real del sistema a lo largo del tiempo.
#   3. Estimamo cuantitativamente métricas clave: la Probabilidad de Extinción (P_ext), el Tiempo Medio hasta la Absorción 
#      E[T_abs] y los Intervalos de Confianza al 95% para analizar la variabilidad estocástica.



######################################################################################################################
#                                                   LIBRERÍAS                                                        #
######################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import os

######################################################################################################################
#                                              PARÁMETROS INICIALES                                                  #
######################################################################################################################
n0 = 1000
pI = 0.05
lambda_descendientes = 1.15
p0 = 0.15
p1 = 0.95
m0 = 0.05
m1 = 0.60

generaciones = 90
num_simulaciones = 500  # número de réplicas Monte Carlo

######################################################################################################################
#                                     MATRICES Y REGISTROS PARA MONTE CARLO                                          #
######################################################################################################################
matriz_totales = np.zeros((num_simulaciones, generaciones + 1))
matriz_enfermos = np.zeros((num_simulaciones, generaciones + 1))
matriz_sanos = np.zeros((num_simulaciones, generaciones + 1))

tiempos_extincion = []

######################################################################################################################
#                                           EXPERIMENTO MONTE CARLO                                                  #
######################################################################################################################
print(f"Ejecutando {num_simulaciones} simulaciones Monte Carlo...")

for sim in range(num_simulaciones):
    np.random.seed(1234 + sim) # cambiar la semilla

    poblacion = np.random.rand(n0) < pI

    hist_tot = [len(poblacion)]
    hist_enf = [np.sum(poblacion)]
    hist_san = [len(poblacion) - np.sum(poblacion)]

    for i in range(1, generaciones + 1):
        if len(poblacion) == 0:
            ceros_restantes = generaciones - i + 1
            hist_tot.extend([0] * ceros_restantes)
            hist_enf.extend([0] * ceros_restantes)
            hist_san.extend([0] * ceros_restantes)
            tiempos_extincion.append(i)
            break

        siguiente_gen = []

        for e in poblacion:
            prob_muerte = m1 if e else m0
            if np.random.rand() < prob_muerte:
                continue  # Muere y no se reproduce

            num_descendientes = np.random.poisson(lambda_descendientes)
            prob_enfermedad_descendiente = p1 if e else p0

            hijos = (
                np.random.rand(num_descendientes) < prob_enfermedad_descendiente
            )
            siguiente_gen.extend(hijos)

        poblacion = np.array(siguiente_gen, dtype=bool)

        n_t = len(poblacion)
        n_e = np.sum(poblacion)
        hist_tot.append(n_t)
        hist_enf.append(n_e)
        hist_san.append(n_t - n_e)

    if len(tiempos_extincion) <= sim:
        tiempos_extincion.append(generaciones)

    # guardar las trayectorias de la réplica en las matrices
    matriz_totales[sim, :] = hist_tot
    matriz_enfermos[sim, :] = hist_enf
    matriz_sanos[sim, :] = hist_san

######################################################################################################################
#                                      CÁLCULOS ESTADÍSTICOS Y RESUMEN                                               #
######################################################################################################################
# Medias poblacionales a lo largo del tiempo
media_totales = np.mean(matriz_totales, axis=0)
media_enfermos = np.mean(matriz_enfermos, axis=0)
media_sanos = np.mean(matriz_sanos, axis=0)

# Intervalo de Confianza al 95% 
ic_inferior = np.percentile(matriz_totales, 2.5, axis=0)
ic_superior = np.percentile(matriz_totales, 97.5, axis=0)

# Métricas de absorción
extinguidas = [t for t in tiempos_extincion if t < generaciones]
prob_extincion = (len(extinguidas) / num_simulaciones) * 100

if extinguidas:
    tiempo_medio_absorcion = np.mean(extinguidas)
    mediana_absorcion = np.median(extinguidas)
    std_absorcion = np.std(extinguidas, ddof=1)
    min_absorcion = np.min(extinguidas)
    max_absorcion = np.max(extinguidas)
    p25_absorcion = np.percentile(extinguidas, 25)
    p75_absorcion = np.percentile(extinguidas, 75)
else:
    tiempo_medio_absorcion = mediana_absorcion = std_absorcion = 0.0
    min_absorcion = max_absorcion = p25_absorcion = p75_absorcion = 0.0

print("\n=======================================================")
print(f"       RESULTADOS MONTE CARLO ({num_simulaciones} RÉPLICAS)")
print("=======================================================")
print(f"1. Régimen de absorción:")
print(f"   - Ratio experimental de extinción (t_max={generaciones}): {prob_extincion:.2f}%")
print(f"   - Trayectorias extinguidas: {len(extinguidas)} / {num_simulaciones}")

print(f"\n2. Distribución del tiempo de absorción (T_abs):")
print(f"   - Media E[T_abs]: {tiempo_medio_absorcion:.2f} generaciones")
print(f"   - Mediana: {mediana_absorcion:.2f} generaciones")
print(f"   - Desviación típica (sigma): +/- {std_absorcion:.2f} generaciones")
print(f"   - Rango: [{min_absorcion}, {max_absorcion}] generaciones")
print(f"   - Percentil 25 (Q1): {p25_absorcion:.2f} generaciones")
print(f"   - Percentil 75 (Q3): {p75_absorcion:.2f} generaciones")
print("=======================================================\n")

######################################################################################################################
#                                           VISUALIZACIÓN MONTE CARLO                                                #
######################################################################################################################
fig, ax = plt.subplots(figsize=(12, 6))
eje_x = np.arange(generaciones + 1)

for sim in range(min(40, num_simulaciones)):
    ax.plot(eje_x, matriz_totales[sim, :], color="gray", alpha=0.15, linewidth=0.8)

ax.fill_between(
    eje_x,
    ic_inferior,
    ic_superior,
    color="black",
    alpha=0.12,
    label="IC 95% (Población Total)",
)

ax.plot(
    eje_x,
    media_totales,
    label="Población Media Total",
    color="black",
    linewidth=2.5,
)
ax.plot(
    eje_x,
    media_enfermos,
    label="Media Enfermos",
    color="crimson",
    linestyle="--",
    linewidth=2,
)
ax.plot(
    eje_x,
    media_sanos,
    label="Media Sanos",
    color="forestgreen",
    linestyle=":",
    linewidth=2,
)

ax.set_title(
    f"Simulación Monte Carlo ({num_simulaciones} réplicas) - Proceso de Galton-Watson",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Generación ($t$)", fontsize=11)
ax.set_ylabel("Número de Individuos ($N_t$)", fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
os.makedirs("./imagenes", exist_ok=True)
plt.savefig("./imagenes/simulacion_incial_sin_farmaco_monte_carlo.png", dpi=300, bbox_inches="tight")
plt.show()