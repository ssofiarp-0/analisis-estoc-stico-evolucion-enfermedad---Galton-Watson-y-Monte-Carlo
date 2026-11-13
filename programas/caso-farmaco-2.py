# Este programa pretende simular la simulacion de una enfermedad con las siguiente características

# PARÁMETROS DEL MODELO BASE:
# n0: tamaño inicial de la población.
# pI: proporción inicial de enfermos respecto a la población inicial (n0).
# lambda_descendientes: parámetro lambda de una distribución Poisson que rige la descendencia de cada progenitor.
# p0: probabilidad de que el sucesor desarrolle la enfermedad si el progenitor está sano (mutación de novo/transmisión).
# p1: probabilidad de que el sucesor desarrolle la enfermedad si el progenitor está enfermo (herencia).
# m0: proporción de mortalidad prematura basal de un individuo sano (causas ajenas a la enfermedad).
# m1: proporción de mortalidad de un individuo enfermo (sin tratamiento/fármaco).
# alpha: eficacia terapéutica del fármaco. 
# rho: tasa de curación de los individuos.

# CONSIDERACIONES DEL PROCESO ESTOCÁSTICO DE GALTON-WATSON:
#   1. La descendencia por progenitor no es constante; sigue una distribución estocástica Poisson(lambda).
#   2. Supervivencia condicional: no todos los individuos enfermos mueren (m1), ni todos los sanos sobreviven (m0).
#   3. La dinámica posee un estado absorbente persistente en N = 0 (extinción total).

# El objetivo de este programa es una única simulación para entender el comportamiento de la mutación junto con el 
# fármaco número uno y su evolución en la población. 



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
alpha = 0.6
rho = 0.35

generaciones = 90
np.random.seed(1234)

######################################################################################################################
#                                       INICIACILACIÓN DE LA POBLACIÓN                                               #
######################################################################################################################

# generacion de la poblacion sana y enferma
poblacion = np.random.rand(n0) < pI

# historiales
hist_totales = [len(poblacion)]
hist_enfermos = [np.sum(poblacion)]
hist_sanos = [len(poblacion) - np.sum(poblacion)]


######################################################################################################################
#                                           SIMULACIÓN GALTON-WATSON                                                 #
######################################################################################################################

for i in range(1, generaciones+1): 
        # si la poblacion se ha extinguido antes de acabar de simular todas las generaciones
        if (len(poblacion) == 0):
                print(f"Población exinta en la generacion {i}")
                hist_totales.extend([0] * (generaciones - i + 1)) 
                hist_enfermos.extend([0] * (generaciones - i + 1))
                hist_sanos.extend([0] * (generaciones - i + 1))
                break

        # simular siguiente generacion de descendientes
        siguiente_gen = []

        for e in poblacion: 
                prob_muerte = m1*(1-alpha) if e else m0
                if np.random.rand() < prob_muerte: 
                        continue # muere y no se reproduce

                if np.random.rand() < rho: e = False # se cura com prob rho

                num_descendientes = np.random.poisson(lambda_descendientes) # numero descendientes

                prob_enfermedad_descendiente = p1 if e else p0 # asignar la probabilidad

                hijos = np.random.rand(num_descendientes) < prob_enfermedad_descendiente # asignar si tiene la enfermedad
                siguiente_gen.extend(hijos)

        poblacion = np.array(siguiente_gen, dtype = bool) # actualizar poblacion con la nueva generacion simulada

        # registrar estadisticas
        n_tot = len(poblacion)
        n_enf = np.sum(poblacion)
        hist_totales.append(n_tot)
        hist_enfermos.append(n_enf)
        hist_sanos.append(n_tot-n_enf)


######################################################################################################################
#                                           VISUALIZACIÓN RESULTADOS                                                 #
######################################################################################################################
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hist_totales, label="Población Total", color="black", linewidth=2)
ax.plot(hist_enfermos, label="Enfermos", color="crimson", linestyle="--", linewidth=2)
ax.plot(hist_sanos, label="Sanos", color="forestgreen", linestyle=":", linewidth=2)

ax.set_title(f"Modelo 2 (Fármaco 2): Proceso de Galton-Watson ($\\alpha={alpha}, \\rho={rho}$)")
ax.set_xlabel("Generación ($t$)")
ax.set_ylabel("Número de Individuos ($N_t$)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()

os.makedirs("./imagenes", exist_ok=True)
plt.savefig("./imagenes/simulacion_farmaco_2.png", dpi=300, bbox_inches="tight")
plt.show()


































