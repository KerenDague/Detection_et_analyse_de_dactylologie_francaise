import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

""" Utilisation de MediaPipe pour extraire les gestes et les transformer en vecteurs directement utilisables par un réseau de neurones.
    Le modèle utilisé est HandLandMarker(stocké dans un fichier handlandmarker.task). """

model_path = 'handlandmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options = base_options,
    running_mode = VisionRunningMode.IMAGE
)

with HandLandmarker.create_from_options(options) as landmarker: