import cv2
import os
import numpy as np
import random

"""
Script d'augmentation de données sur les vidéos.
Pour chaque vidéo, jusqu'à 3 versions augmentées sont générées :
  - miroir horizontal (seulement pour les signes symétriques)
  - variation de luminosité aléatoire
  - rotation légère aléatoire (entre -15° et +15°)

Les vidéos augmentées sont sauvegardées dans le même dossier que les originales,
avec un suffixe pour les identifier.
"""

base_corpus_path = 'corpus_lsf'
VIDEO_EXTENSIONS = {'.mp4', '.mov'}

# Lettres dont le signe est asymétrique :
LETTRES_SANS_MIROIR = {'Z'}


def augmenter_miroir(frame):
    return cv2.flip(frame, 1)


def augmenter_luminosite(frame, facteur=None):
    if facteur is None:
        facteur = random.uniform(1.3, 1.6)
    frame_float = frame.astype(np.float32) * facteur
    return np.clip(frame_float, 0, 255).astype(np.uint8)


def augmenter_rotation(frame, angle=None):
    if angle is None:
        angle = random.choice([-20, 20])
    h, w = frame.shape[:2]
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def creer_video_augmentee(video_path, output_path, transform_fn, **kwargs):
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        print(f"Erreur : Impossible d'ouvrir : {video_path}")
        return

    width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cam.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # On fixe les paramètres aléatoires une suele fois par vidéo pour etre cohérente
    params = {}
    if transform_fn == augmenter_luminosite:
        params['facteur'] = random.uniform(1.2, 1.6)
    elif transform_fn == augmenter_rotation:
        params['angle'] = random.uniform(-15, 15)

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret or frame is None:
            break
        frame_augmente = transform_fn(frame, **params)
        out.write(frame_augmente)

    cam.release()
    out.release()


# Boucle principale
total_crees = 0

for corpus_lettre in os.listdir(base_corpus_path):
    lettre_path = os.path.join(base_corpus_path, corpus_lettre)
    if not os.path.isdir(lettre_path):
        continue

    # Lettre déduite du nom du dossier 
    lettre = corpus_lettre.strip().upper()
    miroir_autorise = lettre not in LETTRES_SANS_MIROIR

    for fichier in os.listdir(lettre_path):
        extension = os.path.splitext(fichier)[1].lower()
        if extension not in VIDEO_EXTENSIONS:
            continue

        # On ne ré-augmente pas des vidéos déjà augmentées
        nom_sans_ext = os.path.splitext(fichier)[0]
        if nom_sans_ext.endswith(('_miroir', '_luminosite', '_rotation')):
            continue

        video_path = os.path.join(lettre_path, fichier)
        print(f"Augmentation de {fichier} (lettre : {lettre})...")

        # 1. Miroir horizontal (ignoré pour les signes asymétriques)
        if miroir_autorise:
            output_miroir = os.path.join(lettre_path, f"{nom_sans_ext}_miroir.mp4")
            creer_video_augmentee(video_path, output_miroir, augmenter_miroir)
            total_crees += 1

        # 2. Variation de luminosité
        output_luminosite = os.path.join(lettre_path, f"{nom_sans_ext}_luminosite.mp4")
        creer_video_augmentee(video_path, output_luminosite, augmenter_luminosite)
        total_crees += 1

        # 3. Rotation légère
        output_rotate = os.path.join(lettre_path, f"{nom_sans_ext}_rotation.mp4")
        creer_video_augmentee(video_path, output_rotate, augmenter_rotation)
        total_crees += 1

print(f"{total_crees} vidéos créées.")