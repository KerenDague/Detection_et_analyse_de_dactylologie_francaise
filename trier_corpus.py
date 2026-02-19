"""
Script pour trier automatiquement les images de l'alphabet LSF dans des dossiers par lettre.

Ce script parcourt les fichiers d'un dossier source, extrait la lettre associée à chaque image
(dernier caractère avant l'extension du fichier), et deplace l'image dans un sous-dossier
correspondant à cette lettre .

"""
import os
import shutil

# Dossier source 
source_dir = "corpus" 

# Dossier de destination 
dest_dir = "corpus_lsf"
os.makedirs(dest_dir, exist_ok=True)

# Parcourir tous les fichiers du dossier source
for filename in os.listdir(source_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Extraire la lettre
        last_char = filename[-5].upper() 
        if last_char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            # Créer le dossier de la lettre si nécessaire
            letter_dir = os.path.join(dest_dir, last_char)
            os.makedirs(letter_dir, exist_ok=True)
            # Déplacer le fichier
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(letter_dir, filename)
            shutil.move(src_path, dest_path)
            print(f"Déplacé : {filename} → {letter_dir}")

print("Finito")
