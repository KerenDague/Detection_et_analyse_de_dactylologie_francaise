import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import os
import numpy as np

"""
Utilisation de MediaPipe pour extraire les gestes et les transformer en vecteurs directement utilisables par un réseau de neurones.
Le modèle utilisé est HandLandMarker(stocké dans un fichier handlandmarker.task).
Ce modèle détermine 21 points sur l'image d'une main pour bien détecter ses mouvements.
Pour chaque vidéo, nous allons donc stocker ces 21 points dans une liste, que nous pourrons ensuite passer à notre réseau de neurones.
Ce script sort une vidéo basée sur la vidéo qu'il a reçue en entrée mais avec les points déterminés par mediapipe (inspiré de https://github.com/prashver/hand-landmark-recognition-using-mediapipe/blob/main/video_input/hand_tracking_video.py)

"""

model_path = 'hand_landmarker.task'
base_corpus_path = 'corpus_lsf_augmente'
output_base_path = 'corpus_augmente_pretraite'

base_options = python.BaseOptions(model_asset_path=model_path)

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

VIDEO_EXTENSIONS = {'.mp4', '.mov'}

options_video = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

def traiter_video(video_path, output_lettre_corpus, video_name):
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        print(f" Erreur : impossible de lire la vidéo : {video_path}")
        return

    width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cam.get(cv2.CAP_PROP_FPS)

    # sécurité fps
    if fps == 0 or fps is None:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name_output = os.path.join(output_lettre_corpus, f"{video_name}_debug.mp4")
    out = cv2.VideoWriter(name_output, fourcc, fps, (width, height))

    try:
        with HandLandmarker.create_from_options(options_video) as landmarker:
            video_data = []
            frame_index = 0  # compteur manuel

            while cam.isOpened():
                ret, frame = cam.read()
                if not ret or frame is None:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # timestamp manuel (FIX PRINCIPAL)
                frame_timestamp_ms = int(frame_index * (1000 / fps))

                try:
                    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                except Exception as e:
                    print(f"[ERREUR MEDIAPIPE] {video_name} frame {frame_index} : {e}")
                    frame_index += 1
                    continue

                frame_points = []

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]

                    for lm in hand_landmarks:
                        x_px = int(lm.x * width)
                        y_px = int(lm.y * height)

                        cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)
                        frame_points.extend([lm.x, lm.y, lm.z])

                # Toujours ajouter une frame (évite les séquences vides)
                if len(frame_points) == 63:
                    video_data.append(frame_points)
                else:
                    video_data.append([0.0] * 63)

                out.write(frame)
                frame_index += 1

            # sécurité finale
            if len(video_data) == 0:
                print(f"[SKIP SAVE] {video_name} — aucune frame valide")
                return

            final_array = np.array(video_data)
            final_path = os.path.join(output_lettre_corpus, f"{video_name}.npy")
            np.save(final_path, final_array)

    except Exception as e:
        print(f"[ERREUR VIDEO] {video_name} : {e}")

    finally:
        cam.release()
        out.release()

    print(f"  Traitement de {video_name} terminé")


# Boucle principale
for corpus_lettre in os.listdir(base_corpus_path):
    lettre_path = os.path.join(base_corpus_path, corpus_lettre)

    output_lettre_corpus = os.path.join(output_base_path, corpus_lettre)
    os.makedirs(output_lettre_corpus, exist_ok=True)

    for fichier in os.listdir(lettre_path):
        fichier_path = os.path.join(lettre_path, fichier)
        extension = os.path.splitext(fichier)[1].lower()
        nom_sans_ext = os.path.splitext(fichier)[0]

        if extension in VIDEO_EXTENSIONS:
            traiter_video(fichier_path, output_lettre_corpus, nom_sans_ext)

        else:
            print(f"[Ignoré] Format non supporté : {fichier}")

cv2.destroyAllWindows()
print("Traitement terminé !")
