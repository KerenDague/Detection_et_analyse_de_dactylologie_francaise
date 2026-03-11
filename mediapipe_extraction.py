import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import os
import numpy as np

""" Utilisation de MediaPipe pour extraire les gestes et les transformer en vecteurs directement utilisables par un réseau de neurones.
    Le modèle utilisé est HandLandMarker(stocké dans un fichier handlandmarker.task).
    Ce modèle détermine 21 points sur l'image d'une main pour bien détecter ses mouvements.
    Pour chaque vidéo, nous allons donc stocker ces 21 points dans une liste, que nous pourrons ensuite passer à notre réseau de neurones.
    Ce script sort une vidéo basée sur la vidéo qu'il a reçue en entrée mais avec les points déterminés par mediapipe (inspiré de https://github.com/prashver/hand-landmark-recognition-using-mediapipe/blob/main/video_input/hand_tracking_video.py) """

model_path = 'LSF/hand_landmarker.task'
base_corpus_path = 'corpus_lsf'
output_base_path = 'corpus_pretraite'

base_options = python.BaseOptions(model_asset_path=model_path)

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options = base_options,
    running_mode = VisionRunningMode.VIDEO,
    num_hands = 1,
)

for corpus_lettre in os.listdir(base_corpus_path):
    lettre_path = os.path.join(base_corpus_path, corpus_lettre)

    # Création des dossiers de sortie par lettre
    output_lettre_corpus = os.path.join(output_base_path, corpus_lettre)
    os.makedirs(output_lettre_corpus, exist_ok=True)

    for video in os.listdir(lettre_path):
        video_path = os.path.join(lettre_path, video)
        cam = cv2.VideoCapture(video_path)

        width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cam.get(cv2.CAP_PROP_FPS)

        # Création de la vidéo de sortie, sur laquelle on rajoutera les points trouvés par mediapipe
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        name_output = os.path.join(output_lettre_corpus, f"{video_name}_debug.mp4")
        out = cv2.VideoWriter(name_output, fourcc, fps, (width, height))


        with HandLandmarker.create_from_options(options) as landmarker:
            video_data = []

            while cam.isOpened():
                ret, frame = cam.read()
                if not ret or frame is None:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                frame_timestamp_ms = int(cam.get(cv2.CAP_PROP_POS_MSEC))
                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                if result.hand_landmarks:
                    poignet = result.hand_landmarks[0][0]
                    print(f"Frame {frame_timestamp_ms}ms : Poignet à x={poignet.x:.2f}, y={poignet.y:.2f}")
                    hand_landmarks = result.hand_landmarks[0]

                    frame_points = []

                    # On parcourt les points de la main; hand_landmarks contient différents points qui correspondent chacun à un endroit de la main
                    for lm in hand_landmarks:
                        x_px = int(lm.x * width)
                        y_px = int(lm.y * height)

                        # On affiche les points trouvés par mediapipe pour vérifier sur la vidéo de sortie
                        cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)

                        frame_points.extend([lm.x, lm.y, lm.z])

                    video_data.append(frame_points)

                out.write(frame)

            final_array = np.array(video_data)
            video_name = os.path.splitext(video)[0]
            final_path = os.path.join(output_lettre_corpus, f"{video_name}.npy")
            np.save(final_path, final_array)

        cam.release()
        out.release()
        print(f'Traitement de {final_path} terminé')

cv2.destroyAllWindows()
print("Traitement terminé !")
