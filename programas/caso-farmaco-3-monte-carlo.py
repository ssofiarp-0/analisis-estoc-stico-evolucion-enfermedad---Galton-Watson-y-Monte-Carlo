import matplotlib.pyplot as plt
import numpy as np
import os
import time

######################################################################################################################
#                                      SELECCIÓN DEL MODELO A SIMULAR                                                #
######################################################################################################################
MODELO_ACTUAL = 3  # <-- Cambia a 2 para Fármaco 2, o a 3 para Fármaco 3

configuraciones = {
    2: {"nombre": "Modelo 2: Fármaco 2", "alpha": 0.60, "rho": 0.35, "archivo": "simulacion_farmaco_2_monte_carlo.png"},
    3: {"nombre": "Modelo 3: Fármaco 3", "alpha": 0.85, "rho": 0.70, "archivo": "simulacion_farmaco_3_monte_carlo.png"},
}

cfg = configuraciones[MODELO_ACTUAL]
alpha = cfg["alpha"]
rho = cfg["rho"]

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
m1_trat = m1 * (1 - alpha)

generaciones = 90
num_simulaciones = 500

######################################################################################################################
#                                     MATRICES Y REGISTROS PARA MONTE CARLO                                          #
######################################################################################################################
matriz_totales = np.zeros((num_simulaciones, generaciones + 1))
matriz_enfermos = np.zeros((num_simulaciones, generaciones + 1))
matriz_sanos = np.zeros((num_simulaciones, generaciones + 1))
tiempos_extincion = []

inicio_tiempo = time.time()
print(f"Ejecutando {num_simulaciones} simulaciones Monte Carlo para {cfg['nombre']} (alpha={alpha}, rho={rho})...")

######################################################################################################################
#                                    EXPERIMENTO MONTE CARLO VECTORIZADO                                             #
######################################################################################################################
for sim in range(num_simulaciones):
    np.random.seed(1234 + sim)

    # Inicialización estocástica en t=0
    enfermos = np.random.binomial(n0, pI)
    sanos = n0 - enfermos

    hist_tot = [n0]
    hist_enf = [enfermos]
    hist_san = [sanos]

    for i in range(1, generaciones + 1):
        if sanos + enfermos == 0:
            ceros_restantes = generaciones - i + 1
            hist_tot.extend([0] * ceros_restantes)
            hist_enf.extend([0] * ceros_restantes)
            hist_san.extend([0] * ceros_restantes)
            tiempos_extincion.append(i)
            break

        # 1. Supervivencia (Binomial equivalente al bucle individuo a individuo)
        sanos_vivos = np.random.binomial(sanos, 1 - m0)
        enfermos_vivos = np.random.binomial(enfermos, 1 - m1_trat)

        # 2. Curación antes de reproducirse con probabilidad rho
        curados = np.random.binomial(enfermos_vivos, rho)
        sanos_rep = sanos_vivos + curados
        enfermos_rep = enfermos_vivos - curados

        # 3. Número total de descendientes (Suma de Poisson = Poisson(N * lambda))
        hijos_de_sanos = np.random.poisson(sanos_rep * lambda_descendientes)
        hijos_de_enfermos = np.random.poisson(enfermos_rep * lambda_descendientes)

        # 4. Herencia y mutación
        enf_desde_sanos = np.random.binomial(hijos_de_sanos, p0)
        sanos_desde_sanos = hijos_de_sanos - enf_desde_sanos

        enf_desde_enfermos = np.random.binomial(hijos_de_enfermos, p1)
        sanos_desde_enfermos = hijos_de_enfermos - enf_desde_enfermos

        sanos = sanos_desde_sanos + sanos_desde_enfermos
        enfermos = enf_desde_sanos + enf_desde_enfermos
        total = sanos + enfermos

        hist_tot.append(total)
        hist_enf.append(enfermos)
        hist_san.append(sanos)

    if len(tiempos_extincion) <= sim:
        tiempos_extincion.append(generaciones + 1)

    matriz_totales[sim, :] = hist_tot
    matriz_enfermos[sim, :] = hist_enf
    matriz_sanos[sim, :] = hist_san

tiempo_total = time.time() - inicio_tiempo

######################################################################################################################
#                                      CÁLCULOS ESTADÍSTICOS Y RESUMEN                                               #
######################################################################################################################
media_totales = np.mean(matriz_totales, axis=0)
media_enfermos = np.mean(matriz_enfermos, axis=0)
media_sanos = np.mean(matriz_sanos, axis=0)

ic_inferior = np.percentile(matriz_totales, 2.5, axis=0)
ic_superior = np.percentile(matriz_totales, 97.5, axis=0)

extinguidas = [t for t in tiempos_extincion if t <= generaciones]
prob_extincion = (len(extinguidas) / num_simulaciones) * 100

print(f"\nCompletado en {tiempo_total:.3f} segundos.")
print("=======================================================")
print(f"  RESULTADOS MONTE CARLO - {cfg['nombre'].upper()}")
print("=======================================================")
print(f"1. Régimen de absorción:")
print(f"   - Ratio experimental de extinción (t_max={generaciones}): {prob_extincion:.2f}%")
print(f"   - Trayectorias extinguidas: {len(extinguidas)} / {num_simulaciones}")

print(f"\n2. Crecimiento poblacional al final (t={generaciones}):")
print(f"   - Población media total E[N_90]: {media_totales[-1]:,.2f} individuos")
print(f"   - Desviación típica (sigma): +/- {np.std(matriz_totales[:, -1], ddof=1):,.2f}")
print(f"   - Mediana Total: {np.median(matriz_totales[:, -1]):,.2f}")
print(f"   - IC 95% Total: [{ic_inferior[-1]:,.2f}, {ic_superior[-1]:,.2f}]")
print(f"   - Media Sanos E[S_90]: {media_sanos[-1]:,.2f} ({(media_sanos[-1]/media_totales[-1])*100:.2f}%)")
print(f"   - Media Enfermos E[I_90]: {media_enfermos[-1]:,.2f} ({(media_enfermos[-1]/media_totales[-1])*100:.2f}%)")
print("=======================================================\n")

######################################################################################################################
#                                           VISUALIZACIÓN MONTE CARLO                                                #
######################################################################################################################
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
    f"Simulación Monte Carlo ({num_simulaciones} réplicas) - Modelo 3: Fármaco 3 ($\\alpha={alpha}, \\rho={rho}$)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Generación ($t$)", fontsize=11)
ax.set_ylabel("Número de Individuos ($N_t$)", fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
os.makedirs("./imagenes", exist_ok=True)
plt.savefig("./imagenes/simulacion_farmaco_3_monte_carlo.png", dpi=300, bbox_inches="tight")
plt.show()  