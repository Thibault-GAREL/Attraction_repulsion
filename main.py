from numba import cuda
import numpy as np
import pygame
import sys
import random
import math

# Initialisation de Pygame
pygame.init()

# Dimensions de la fenêtre
LARGEUR, HAUTEUR = 800, 600

# Couleurs
BLANC = (255, 255, 255)
ROUGE = (255, 0, 0)
BLEU = (0, 0, 255)
VERT = (0, 255, 0)
BLEU_SOMBRE = (20, 20, 40)

couleur = [ROUGE, VERT, BLEU]

nb_element = [500, 500, 3500]
rayon = 2

coef_friction = 0.5

multiplicateur = 1/50
attraction_coef = np.array([[3, -3, 0],
                            [3, 1, 0],
                            [-1, -1, -0.5]], dtype=np.float32)

# Paramètres du GPU
threads_per_block = 256

blocks_per_grid = []

positions = []
vitesses = []

d_positions = []
d_vitesses = []

# Création de la fenêtre
ecran = pygame.display.set_mode((LARGEUR, HAUTEUR))
pygame.display.set_caption("Attraction Répultion")


# Fonctions GPU
@cuda.jit
def calcul_attraction_gpu(self_positions, self_vitesses, other_positions, other_vitesses, self_nb_element, other_nb_element, attraction_coeff):
    i = cuda.grid(1)  # Identifie le thread
    if i < self_nb_element:
        px, py = self_positions[i, 0], self_positions[i, 1]
        vx, vy = self_vitesses[i, 0], self_vitesses[i, 1]

        for j in range(other_nb_element):
            if i != j:  # Ne pas s'attirer soi-même
                dx = other_positions[j, 0] - px
                dy = other_positions[j, 1] - py

                # Gestion des bords (torique)
                if dx > LARGEUR / 2:
                    dx -= LARGEUR
                elif dx < -LARGEUR / 2:
                    dx += LARGEUR

                if dy > HAUTEUR / 2:
                    dy -= HAUTEUR
                elif dy < -HAUTEUR / 2:
                    dy += HAUTEUR

                distance = math.sqrt(dx ** 2 + dy ** 2)
                # if distance > (rayon * 2) and distance < 100:
                if distance > 0 and distance < 100:
                    force_x = (dx / distance) * attraction_coeff
                    force_y = (dy / distance) * attraction_coeff
                    vx += force_x
                    vy += force_y

                # if distance <= (rayon * 2) and distance != 0:
                #     overlap = 2 * rayon - distance
                #     correction_x = (dx / distance) * overlap
                #     correction_y = (dy / distance) * overlap
                #
                #     # Repousse les billes de moitié dans des directions opposées
                #     px -= correction_x / 2
                #     py -= correction_y / 2
                #
                #     other_positions[j, 0] += correction_x / 2
                #     other_positions[j, 1] += correction_y / 2
                #
                #     # vx *= 0
                #     # vy *= 0

        # Appliquer la vitesse calculée et ajouter une friction
        self_vitesses[i, 0] = vx * coef_friction
        self_vitesses[i, 1] = vy * coef_friction

@cuda.jit
def deplacer_gpu(positions, vitesses, nb_particules):
    i = cuda.grid(1)
    if i < nb_particules:
        # Mettre à jour les positions
        positions[i, 0] += vitesses[i, 0]
        positions[i, 1] += vitesses[i, 1]

        # Gestion des bords (torique)
        if positions[i, 0] < 0:
            positions[i, 0] += LARGEUR
        elif positions[i, 0] >= LARGEUR:
            positions[i, 0] -= LARGEUR

        if positions[i, 1] < 0:
            positions[i, 1] += HAUTEUR
        elif positions[i, 1] >= HAUTEUR:
            positions[i, 1] -= HAUTEUR

        distance = math.sqrt(positions[i, 0] ** 2 + positions[i, 1] ** 2)
        if distance < 2 * rayon:
            overlap = 2 * rayon - distance
            positions[i, 0] -= overlap * (positions[i, 0] / distance)
            positions[i, 1] -= overlap * (positions[i, 1] / distance)

# Initialisation des données
for nb in range(len(nb_element)):
    positions_interne = np.array([[random.randint(5, LARGEUR - 5), random.randint(5, HAUTEUR - 5)]
                      for _ in range(nb_element[0])], dtype=np.float32)
    positions.append(positions_interne)
    vitesses_interne = np.zeros((nb_element[0], 2), dtype=np.float32)
    vitesses.append(vitesses_interne)

    d_positions_interne = cuda.to_device(positions[nb])
    d_positions.append(d_positions_interne)

    d_vitesses_interne = cuda.to_device(vitesses[nb])
    d_vitesses.append(d_positions_interne)

    blocks_per_grid_interne = (nb_element[nb] + threads_per_block - 1) // threads_per_block
    blocks_per_grid.append(blocks_per_grid_interne)



# Boucle principale
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Effacer l'écran
    ecran.fill(BLEU_SOMBRE)

    # Calcul sur GPU
    for nb in range(len(nb_element)):
        for chaque_couleur in range (len(nb_element)):
            calcul_attraction_gpu[blocks_per_grid[nb], threads_per_block](
                d_positions[nb], d_vitesses[nb], d_positions[chaque_couleur], d_vitesses[chaque_couleur], nb_element[nb], nb_element[chaque_couleur], attraction_coef[nb][chaque_couleur] * multiplicateur)
            deplacer_gpu[blocks_per_grid[nb], threads_per_block](
                d_positions[nb], d_vitesses[nb], nb_element[nb])

        positions_c = d_positions[nb].copy_to_host()
        # print("Positions GPU (host):", positions_c)
        for x, y in positions_c:
            pygame.draw.circle(ecran, couleur[nb], (int(x), int(y)), rayon)


    # Mettre à jour l'affichage
    pygame.display.flip()

pygame.quit()
sys.exit()