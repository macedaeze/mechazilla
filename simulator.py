from Pynite import FEModel3D
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.widgets import Slider
import pyvista as pv

# Crear nuevo modelo de elemento finito
truss = FEModel3D()

# --- NODOS ---
truss.add_node('N0', 0, 0, 0)
truss.add_node('N1', 4.813, 5.374, 0)
truss.add_node('N2', 9.285, 9.374, 0)
truss.add_node('N3', 11.285, 9.374, 0)
truss.add_node('N4', 15.285, 9.375, 0)
truss.add_node('N5', 19.285, 9.374, 0)
truss.add_node('N6', 23.285, 9.374, 0)
truss.add_node('N7', 27.285, 9.374, 0)
truss.add_node('N8', 31.285, 9.374, 0)
truss.add_node('N9', 31.285, 5.374, 0)
truss.add_node('N10', 27.285, 5.374, 0)
truss.add_node('N11', 23.285, 5.374, 0)
truss.add_node('N12', 19.285, 5.374, 0)
truss.add_node('N13', 15.285, 5.374, 0)
truss.add_node('N14', 11.285, 5.374, 0)
truss.add_node('N15', 9.285, 5.374, 0)
truss.add_node('N16', 5.809, 1.5, 0)
truss.add_node('N17', 31.285, 5.374, -4)
truss.add_node('N18', 27.285, 5.374, -4)
truss.add_node('N19', 23.285, 5.374, -4)
truss.add_node('N20', 19.285, 5.374, -4)
truss.add_node('N21', 15.285, 5.374, -4)
truss.add_node('N22', 11.285, 5.374, -4)
truss.add_node('N23', 9.285, 5.374, -5)
truss.add_node('N24', 0, 0, -16)

# --- MATERIAL PRELIMINAR ---
E = 200e9   # Módulo Elástico, en Pa
rho = 7850  # densidad, en kg/m^3
G = 79.3e9  # Módulo de Corte, en Pa
nu = 0.3 # Coeficiente de Poisson, adimensional
truss.add_material('Steel', E, G, nu, rho)

# --- SECCIÓN ---
A = 0.0925  # Área de sección, en m^2 
Iy = 1e-4  # Momento de Inercia en eje y local, en m^4
Iz = 1e-4  # Momento de inercia en eje z local, en m^4
J = 1e-4   # Momento polar de inercia, en m^4
truss.add_section('TrussSection', A, Iy, Iz, J)

# --- BARRAS ---
bar_connectivity = [
    [0,1], [1,2], [2,3],
    [3,4], [4,5], [5,6],
    [6,7], [7,8], [8,9],
    [9,10], [10,11], [11,12],
    [12,13], [13,14], [14,15],
    [15,16], [0,16], [16,1],
    [1,15], [15,2], [15,3],
    [3,14], [14,4], [4,13],
    [13,5], [5,12], [12,6],
    [6,11], [11,7], [7,10],
    [10,8], [0,24], [24,16],
    [24,1], [16,23], [1,23],
    [15,23], [2,23], [15, 22], 
    [3,22], [14,22], [14,21], 
    [4,21], [13,21], [13,20], 
    [5,20], [12,20], [12,19], 
    [6,19], [11,19], [11,18], 
    [7,18], [10,18], [10,17], 
    [8,17], [9,17], [24,23],
    [23,22], [22,21], [21,20],
    [20,19], [19,18], [18,17]
]

for i, (i1, i2) in enumerate(bar_connectivity):
    truss.add_member(f'B{i}', f'N{i1}', f'N{i2}', 'Steel', 'TrussSection')
    # Liberar los momentos en los extremos de los miembros para convertirlos en barras de reticulado 
    truss.def_releases(f'B{i}', False, False, False, False, True, True, \
                           False, False, False, False, True, True)

# --- CONDICIONES DE APOYO ---
# Nodo 0: Ux, Uy
truss.def_support('N0', True, True, False, False, False, False)
# Nodo 24: Ux, Uy, Uz
truss.def_support('N24', True, True, True, False, False, False)
# Nodo 8: Uy
truss.def_support('N8', False, True, False, False, False, False)

# --- CARGAS ---
# Cargas en Nodo 11 
truss.add_node_load('N11', 'FZ', -2697000)
truss.add_node_load('N11', 'FY', 337000)

# --- ANÁLISIS ---
truss.analyze(check_statics=True)

# --- RESULTADOS NUMÉRICOS ---
results_forces = []
results_stresses = []

for m in truss.members.values():
    # Fuerza axial (positivo = tension, negativo = compresion)
    axial = m.max_axial()

    # Acceso a sección para calcular esfuerzo
    A = m.section.A
    stress = axial / A
    state = "Tensión" if axial > 0 else ("Fuerza 0" if axial == 0 else "Compresión")
    results_forces.append([m.name, axial, state])
    results_stresses.append([m.name, stress, state])

df1 = pd.DataFrame(results_forces, columns=["Barra", "F_axial [N]", "Estado"])
df2 = pd.DataFrame(results_stresses, columns=["Barra", "σ [Pa]", "Estado"])

print("\n=== REACCIONES EN LOS VINCULOS ===")
print("Nodo 0")
print(truss.nodes['N0'].RxnFX) # .RxnFY, .RxnFZ, .RxnMX, .RxnMY, .RxnMZ
print(truss.nodes['N0'].RxnFY)
print(truss.nodes['N0'].RxnFZ) 
print("Nodo 24")
print(truss.nodes['N24'].RxnFX) # .RxnFY, .RxnFZ, .RxnMX, .RxnMY, .RxnMZ
print(truss.nodes['N24'].RxnFY)
print(truss.nodes['N24'].RxnFZ) 
print("Nodo 8")
print(truss.nodes['N8'].RxnFX) # .RxnFY, .RxnFZ, .RxnMX, .RxnMY, .RxnMZ
print(truss.nodes['N8'].RxnFY)
print(truss.nodes['N8'].RxnFZ) 
print("\n")

print("\n=== RESULTADOS DE FUERZAS AXIALES POR BARRA ===\n")
print(df1.to_string(index=False))

print("\n\n=== RESULTADOS DE ESFUERZOS POR BARRA ===\n")
print(df2.to_string(index=False))

# Exportar tabla a CSV (opcional)
df1.to_csv("resultados_fuerzas.csv", index=False)
df1.to_csv("resultados_esfuerzos.csv", index=False)
print("\nResultados guardados")

# ============================================================
# TABLA DE ÁREAS MÍNIMAS VIABLES Y VERIFICACIÓN DE CUMPLIMIENTO
# ============================================================
# --- MATERIAL: Acero estructural S355 (valores representativos) ---

E = 210e9       # Módulo de Young (Pa) -- típico según EN10025 / Eurocode
rho = 7850      # Densidad (kg/m^3)
sigma_y = 355e6 # Límite elástico (Pa) - 355 MPa
sigma_u = 470e6 # Resistencia última a tracción (Pa) - 470 MPa mínimo garantizado
nu = 0.30       # Coeficiente de Poisson (adimensional)
G = E / (2 * (1 + nu))  # Módulo de corte (Pa), calculado
sigma_adm_t = 237e6   # Pa (tensión)
sigma_adm_c = 118e6   # Pa (compresión)

A_actual = A  # área real usada en el modelo (m^2)

tabla = []

for m in truss.members.values():
    axial = m.max_axial()  # fuerza axial en N
    estado = "Tensión" if axial > 0 else ("Fuerza 0" if axial == 0 else "Compresión")

    # Selección del sigma admisible según el estado
    if estado == "Tensión":
        sigma_adm = sigma_adm_t
    elif estado == "Compresión":
        sigma_adm = sigma_adm_c
    else:
        sigma_adm = 1e9  # arbitrario ya que A_min = 0 en fuerza 0

    # Área mínima viable
    if axial == 0:
        A_min = 0.0
    else:
        A_min = abs(axial) / sigma_adm

    cumple = "Sí" if A_actual >= A_min else "No"

    tabla.append([m.name, estado, A_min, cumple])

dfA = pd.DataFrame(tabla, columns=["Barra", "Estado", "A_min [m²]", "Cumple"])

print("\n=========== VERIFICACIÓN DE ÁREA MÍNIMA ===========\n")
print(dfA.to_string(index=False))
print("\nÁrea actual utilizada en el modelo: A = {:.6f} m²".format(A_actual))

# ============================================================
# CÁLCULO DEL ESPESOR t PARA CHS Ø700 mm SEGÚN A_min
# ============================================================

import math

D_o = 0.7  # diámetro exterior del tubo en metros
espesores_comerciales = [0.006, 0.008, 0.010, 0.0125, 0.015, 0.020, 0.025, 0.030]  # en metros

resultados_t = []

for idx, row in dfA.iterrows():
    barra = row["Barra"]
    A_min = row["A_min [m²]"]

    # Si no hay fuerza, no hace falta t
    if A_min == 0:
        resultados_t.append([barra, A_min, 0.0, "Sin requerimiento"])
        continue

    # Ecuación: A = π(D_o t - t²)
    # t² - D_o t + (A/π) = 0
    a = 1
    b = -D_o
    c = A_min / math.pi

    disc = b**2 - 4*a*c

    if disc <= 0:
        resultados_t.append([barra, A_min, None, "No existe t real – revisar fuerza"])
        continue

    # Solución física válida
    t1 = (D_o - math.sqrt(disc)) / 2   # m
    t_calc = t1

    # Seleccionar espesor comercial inmediato superior
    t_prop = None
    for t_std in espesores_comerciales:
        if t_std >= t_calc:
            t_prop = t_std
            break

    if t_prop is None:
        perfil = "Espesor > 30 mm – usar sección caja"
    else:
        perfil = f"CHS 700 × {t_prop*1000:.1f} mm"

    resultados_t.append([
        barra,
        A_min,
        t_calc * 1000,   # mm
        perfil
    ])

# Convertir a DataFrame
df_t = pd.DataFrame(resultados_t,
                    columns=["Barra", "A_min [m²]", "t_calc [mm]", "Perfil sugerido"])

print("\n=========== ESPESOR REQUERIDO PARA CHS Ø700 mm ===========\n")
print(df_t.to_string(index=False))


# ============================================================
# CÁLCULO DE DEFORMACIÓN EN LA BARRA MÁS CRÍTICA (MÁX |σ|)
# ============================================================

# Tomamos el área real usada en el modelo (A_actual) y el material del modelo
E_model = 210e9 

# Buscar la barra con mayor esfuerzo absoluto
crit_bar = None
crit_sigma = 0.0

for m in truss.members.values():
    axial = m.max_axial()      # fuerza axial en N (tensión +, compresión -)
    sigma = axial / 0.063146   # esfuerzo en Pa, usando el área actual
    if abs(sigma) > abs(crit_sigma):
        crit_sigma = sigma
        crit_bar = m

# Obtener longitud geométrica de la barra crítica a partir de sus nodos
i_node = crit_bar.i_node
j_node = crit_bar.j_node

# En algunas versiones de PyNite, i_node / j_node pueden ser strings con el nombre
from collections.abc import Hashable
if isinstance(i_node, Hashable) and isinstance(i_node, str):
    i_node = truss.nodes[i_node]
if isinstance(j_node, Hashable) and isinstance(j_node, str):
    j_node = truss.nodes[j_node]

L_crit = ((j_node.X - i_node.X)**2 +
          (j_node.Y - i_node.Y)**2 +
          (j_node.Z - i_node.Z)**2) ** 0.5

# Deformación unitaria y deformación total
eps_crit = crit_sigma / E_model          # deformación unitaria (adimensional)
delta_crit = eps_crit * L_crit           # deformación total (m)

print("\n=========== DEFORMACIÓN EN BARRA CRÍTICA ===========\n")
print(f"Barra crítica: {crit_bar.name}")
print(f"Esfuerzo σ = {crit_sigma:.3e} Pa")
print(f"Longitud L = {L_crit:.3f} m")
print(f"Deformación unitaria ε = {eps_crit:.3e}")
print(f"Deformación total δ = {delta_crit*1000:.3f} mm")



# ============================================================
# DESPLAZAMIENTO EN NODO 11 (Z) POR CASTIGLIANO - FORMULA Σ(Ni^2 Li / (E A P))
# ============================================================

# Datos del material y sección del MODELO (los mismos que usaste al definir 'Steel')
E_model = 210e9      # Pa
A_model = 0.063146          # m² (la A que usaste al crear 'TrussSection')

# Carga vertical aplicada en el nodo 11 (módulo, en N)
P_v = 2697000.0      # N

delta_11_z_cast = 0.0  # acumulador

for m in truss.members.values():
    # Fuerza axial en la barra i (N_i)
    Ni = m.max_axial()  # N

    # Longitud de la barra i (Li)
    i_node = m.i_node
    j_node = m.j_node
    dx = j_node.X - i_node.X
    dy = j_node.Y - i_node.Y
    dz = j_node.Z - i_node.Z
    Li = (dx**2 + dy**2 + dz**2) ** 0.5  # m

    # Aporte de la barra i a la suma: Ni^2 * Li / (E * A * P)
    delta_11_z_cast += (Ni**2 * Li) / (E_model * A_model * P_v)

print("\n=========== DESPLAZAMIENTO EN NODO 11 (Z) POR CASTIGLIANO (Σ N_i² L_i / E A P) ===========\n")
print(f"δ_11,z = {delta_11_z_cast:.6e} m")
print(f"δ_11,z = {delta_11_z_cast*1000:.6f} mm")


# (Opcional) Comparar con el desplazamiento que da directamente PyNite
print("\n--- COMPARACIÓN CON DESPLAZAMIENTO FEM (PyNite) ---")

node11 = truss.nodes['N11']
dz_raw = node11.DZ  # puede ser dict o float según versión

# Si es un dict (desplazamientos por combinación de carga), tomo el primero
if isinstance(dz_raw, dict):
    # Por ejemplo 'Combo 1'. Tomo el primer valor del diccionario.
    dz_val = list(dz_raw.values())[0]
else:
    dz_val = dz_raw  # ya es un número

print(f"DZ en N11 segun modelo original = {dz_val:.6e} m")
print(f"DZ en N11 segun modelo original = {dz_val*1000:.6f} mm")

