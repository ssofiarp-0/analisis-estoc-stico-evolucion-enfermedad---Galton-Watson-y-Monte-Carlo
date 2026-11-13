# Análisis Estocástico de la Evolución de Patologías mediante Procesos de Galton-Watson

**Autora:** Sofía Rodríguez Peláez  
**Estudios:** Doble Grado en Ingeniería Informática y Estadística — Universidad de Valladolid

---

## Descripción

Este proyecto estudia la evolución de una enfermedad hereditaria en una población finita y el impacto de tres tratamientos farmacológicos diferentes. Para ello, se modela la población mediante un **Proceso de Ramificación de Galton-Watson** en tiempo discreto y se realizan simulaciones de Monte Carlo ($M = 500$ réplicas) en Python para estimar la probabilidad de extinción y el crecimiento poblacional a lo largo de $90$ generaciones.

---

## Escenarios Evaluados

Se analiza un escenario sin tratamiento (Caso Base) y tres fármacos definidos por su eficacia para reducir la mortalidad ($\alpha$) y su tasa de curación ($\rho$):

* **Caso Base ($\alpha = 0.00, \, \rho = 0.00$):** La población converge inevitablemente a la extinción ($\mathcal{R}_0^{\text{global}} = 0.9362 < 1$).
* **Fármaco 1 ($\alpha = 0.30, \, \rho = 0.10$):** Retrasa el colapso poblacional, pero no evita la extinción a largo plazo ($\mathcal{R}_0^{\text{global}} = 0.9655 < 1$).
* **Fármaco 2 ($\alpha = 0.60, \, \rho = 0.35$):** Evita la extinción ($\mathcal{R}_0^{\text{global}} = 1.0340 > 1$), aunque la enfermedad persiste de forma endémica ($\approx 27\%$ de enfermos).
* **Fármaco 3 ($\alpha = 0.85, \, \rho = 0.70$):** Garantiza el crecimiento sostenido de la población y reduce la prevalencia de la enfermedad por debajo del $20\%$ ($\mathcal{R}_0^{\text{global}} = 1.0835 > 1$).

---

## Estructura del Proyecto

* `codigo/`: Scripts en Python con las simulaciones puntuales y los experimentos de Monte Carlo vectorizados.
* `imagenes/`: Gráficas generadas con la evolución temporal de sanos, enfermos y población total.
* `latex/`: Código fuente del documento científico elaborado con la plantilla de Springer.