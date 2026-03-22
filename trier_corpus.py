"""
Script pour trier automatiquement les images de l'alphabet LSF dans des dossiers par lettre.

Ce script parcourt les fichiers d'un dossier source, extrait la lettre associée à chaque image
(dernier caractère avant l'extension du fichier), et deplace l'image dans un sous-dossier
correspondant à cette lettre .

"""
import os
import shutil

# Dossier source 
source_dir = "corpus_entier"

# Dossier de destination 
dest_dir = "corpus_lsf"
os.makedirs(dest_dir, exist_ok=True)

# Parcourir tous les fichiers du dossier source
for root, dirs, files in os.walk(source_dir):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.mov')):
            # Extraire la lettre
            last_char = filename[-5].upper()
            if last_char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                # Créer le dossier de la lettre si nécessaire
                letter_dir = os.path.join(dest_dir, last_char)
                os.makedirs(letter_dir, exist_ok=True)
                # Déplacer le fichier
                src_path = os.path.join(root, filename)
                dest_path = os.path.join(letter_dir, filename)
                # Gérer les doublons de noms de fichiers
                if os.path.exists(dest_path):
                    base = os.path.splitext(filename)[0]
                    ext = os.path.splitext(filename)[1]
                    dest_path = os.path.join(letter_dir, f"{base}_{root.split(os.sep)[-1]}{ext}")
                shutil.copy2(src_path, dest_path)
                print(f"Copié : {filename} → {letter_dir}")

print("Finito")
