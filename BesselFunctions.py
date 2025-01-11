import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn

# Définition des paramètres
x = np.linspace(0, 20, 400)  # Intervalle de x
orders = [0, 1, 2, 3]  # Ordres des fonctions de Bessel à tracer

# Création de la figure et des sous-graphiques
plt.figure(figsize=(10, 6))

for n in orders:
    y = jn(n, x)  # Calcul de la fonction de Bessel de première espèce d'ordre n
    plt.plot(x, y, label=f'J_{n}(x)')

# Configuration des axes et de la légende
plt.title('Fonctions de Bessel de première espèce')
plt.xlabel('x')
plt.ylabel('J_n(x)')
plt.legend()
plt.grid(True)

# Affichage de la figure
plt.show()