import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

"""
Ce script :
Parcourt un dossier de données prétraitées (contenant des fichiers .npy) et calcule les statistiques suivantes pour chaque séquence :
   - Nombre total de vidéos
   - Longueur moyenne des séquences
   - Longueurs minimale et maximale des séquences
   - Liste des fichiers problématiques (shape[0] == 0, aucune main détectée)
Filtre les frames des séquences trop longues en supprimant celles qui s'éloignent le plus du vecteur médian, jusqu'à atteindre 110 frames.
Crée un Dataset PyTorch (LSFDataset) en paddant/tronquant les séquences à la longueur maximale calculée via la méthode IQR, et enregistre :
   - les fichiers .npy paddés dans un dossier par lettre (corpus_pretraite_padded/)
   - l'ensemble du dataset (tenseurs + labels + métadonnées) dans un fichier .pt (dataset_lsf.pt)
"""

# Analyse statistique des séquences

output_base_path = 'corpus_augmente_pretraite'
sequence_lengths = []
fichiers = []

for corpus_lettre in os.listdir(output_base_path):
    lettre_path = os.path.join(output_base_path, corpus_lettre)
    if not os.path.isdir(lettre_path):
        continue
    for fichier in os.listdir(lettre_path):
        if fichier.endswith('.npy'):
            fichier_path = os.path.join(lettre_path, fichier)
            data = np.load(fichier_path, allow_pickle=True)
            sequence_lengths.append(data.shape[0])
            fichiers.append(fichier_path)

sequence_lengths = np.array(sequence_lengths)

print(f"Nombre total de vidéos : {len(sequence_lengths)}")

# Détection des aberrants par méthode IQR
Q1 = np.percentile(sequence_lengths, 25)
Q3 = np.percentile(sequence_lengths, 75)
IQR = Q3 - Q1
borne_basse = Q1 - 1.5 * IQR
borne_haute = Q3 + 1.5 * IQR

print(f"Q1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f}")
print(f"Bornes : [{borne_basse:.0f} frames, {borne_haute:.0f} frames]")

fichiers_aberrants = [
    (fichiers[i], sequence_lengths[i])
    for i in range(len(fichiers))
    if sequence_lengths[i] < borne_basse or sequence_lengths[i] > borne_haute
]

plt.figure(figsize=(10, 6))
plt.hist(sequence_lengths, bins=30, edgecolor='#C71585', color='#FFC0CB')
plt.axvline(np.mean(sequence_lengths), color='#FF69B4', linestyle='dashed', linewidth=1, label=f'Moyenne: {np.mean(sequence_lengths):.1f}')
plt.axvline(np.median(sequence_lengths), color='#C71585', linestyle='dashed', linewidth=1, label=f'Médiane: {np.median(sequence_lengths):.1f}')
plt.xlabel('Nombre de frames par vidéo')
plt.ylabel('Nombre de vidéos')
plt.title('Distribution des longueurs des séquences (avant filtrage)')
plt.legend()
plt.grid(True, color='#FFB6C1', alpha=0.4)
plt.tight_layout()
plt.show()

print(f"Nombre total de fichiers : {len(sequence_lengths)}")
print(f"Longueur moyenne : {np.mean(sequence_lengths):.1f} frames")
print(f"Longueur médiane : {np.median(sequence_lengths):.1f} frames")
print(f"Longueur min/max : {np.min(sequence_lengths)}/{np.max(sequence_lengths)} frames")

print(f"Nombre de fichiers aberrants : {len(fichiers_aberrants)}")
if fichiers_aberrants:
    print("Fichiers concernés :")
    for chemin, longueur in sorted(fichiers_aberrants, key=lambda x: x[1]):
        print(f"  - {chemin}  ({longueur} frames)")
else:
    print("Aucun fichier aberrant trouvé.")

longueur_max = int(borne_haute)
TARGET_FRAMES = 110
print(f"\nlongueur_max retenue : {longueur_max} frames")
print(f"Cible de filtrage : {TARGET_FRAMES} frames")



# Filtrage des frames inutiles

def filtrer_frames(sequence: np.ndarray, target: int) -> np.ndarray:
    """
    Supprime les frames les plus éloignées du vecteur médian
    jusqu'à atteindre target frames. L'ordre chronologique est préservé.
    """
    if len(sequence) <= target:
        return sequence
    median_frame = np.median(sequence, axis=0)
    distances = np.linalg.norm(sequence - median_frame, axis=1)
    indices_gardes = np.argsort(distances)[:target]
    indices_gardes = np.sort(indices_gardes)
    return sequence[indices_gardes]


total_filtrees = 0
for corpus_lettre in os.listdir(output_base_path):
    lettre_path = os.path.join(output_base_path, corpus_lettre)
    if not os.path.isdir(lettre_path):
        continue
    for fichier in os.listdir(lettre_path):
        if not fichier.endswith('.npy'):
            continue
        npy_path = os.path.join(lettre_path, fichier)
        sequence = np.load(npy_path, allow_pickle=True)
        if len(sequence) == 0:
            print(f"[Ignoré] {fichier} — aucune main détectée")
            continue
        avant = len(sequence)
        sequence_filtree = filtrer_frames(sequence, TARGET_FRAMES)
        apres = len(sequence_filtree)
        if avant != apres:
            np.save(npy_path, sequence_filtree)
            total_filtrees += 1
            print(f"{fichier} : {avant} → {apres} frames")

print(f"\nFiltrage terminé — {total_filtrees} fichiers modifiés.")

# Histogramme après filtrage
sequence_lengths_apres = []
for corpus_lettre in os.listdir(output_base_path):
    lettre_path = os.path.join(output_base_path, corpus_lettre)
    if not os.path.isdir(lettre_path):
        continue
    for fichier in os.listdir(lettre_path):
        if fichier.endswith('.npy'):
            data = np.load(os.path.join(lettre_path, fichier), allow_pickle=True)
            sequence_lengths_apres.append(data.shape[0])

sequence_lengths_apres = np.array(sequence_lengths_apres)
print(f"\nNombre total de vidéos après filtrage : {len(sequence_lengths_apres)}")

plt.figure(figsize=(10, 6))
plt.hist(sequence_lengths_apres, bins=30, edgecolor='black', color='pink')
plt.axvline(np.mean(sequence_lengths_apres), color='#FF1493', linestyle='dashed', linewidth=1, label=f'Moyenne: {np.mean(sequence_lengths_apres):.1f}')
plt.axvline(np.median(sequence_lengths_apres), color='#C71585', linestyle='dashed', linewidth=1, label=f'Médiane: {np.median(sequence_lengths_apres):.1f}')
plt.xlabel('Nombre de frames par vidéo')
plt.ylabel('Nombre de vidéos')
plt.title('Distribution des longueurs des séquences (après filtrage)')
plt.legend()
plt.grid(True, color='#FFB6C1', alpha=0.4)
plt.tight_layout()
plt.show()


# Création du Dataset PyTorch

class LSFDataset(Dataset):
    def __init__(self, base_path, letters, longueur_max, output_path=None):
        self.samples = []
        self.letter_map = {letter: num for num, letter in enumerate(letters)}
        self.output_path = output_path
        self.longueur_max = longueur_max

        for letter in letters:
            npy_path = os.path.join(base_path, letter)
            if output_path:
                os.makedirs(os.path.join(output_path, letter), exist_ok=True)
            for file in os.listdir(npy_path):
                if file.endswith('.npy'):
                    self.samples.append((os.path.join(npy_path, file), self.letter_map[letter]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            data = np.load(path)
        except Exception as e:
            print(f"Erreur sur {path} : {e}")
            data = np.zeros((self.longueur_max, 63), dtype=np.float32)

        if len(data) > self.longueur_max:
            data = data[:self.longueur_max]
        elif len(data) < self.longueur_max:
            pad_width = self.longueur_max - len(data)
            padding = np.zeros((pad_width, 63))
            data = np.vstack((data, padding))

        if self.output_path:
            new_path = os.path.join(self.output_path, os.path.basename(os.path.dirname(path)), os.path.basename(path))
            np.save(new_path, data)

        return torch.from_numpy(data).float(), torch.tensor(label)


base_path = "corpus_augmente_pretraite"
output_path = "corpus_augmente_pretraite_padded"
dataset_save_path = "dataset_augmente_test.pt"
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

dataset = LSFDataset(base_path, letters, longueur_max=longueur_max, output_path=output_path)
print(f"\nDataset créé : {len(dataset)} échantillons")
print(f"Nombre total de vidéos dans le dataset : {len(dataset)}")

all_data, all_labels = [], []
for i in range(len(dataset)):
    x, y = dataset[i]
    all_data.append(x)
    all_labels.append(y)

all_data   = torch.stack(all_data)
all_labels = torch.stack(all_labels)

torch.save({"data": all_data, "labels": all_labels, "letter_map": dataset.letter_map, "longueur_max": longueur_max}, dataset_save_path)

print(f"Dataset sauvegardé → {dataset_save_path}  ({all_data.shape[0]} échantillons, shape tenseur : {tuple(all_data.shape)})")
print(f"Fichiers .npy paddés enregistrés dans → {os.path.abspath(output_path)}/")
