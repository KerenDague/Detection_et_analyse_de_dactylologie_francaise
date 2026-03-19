import os
from moviepy import VideoFileClip, ImageSequenceClip
from rembg import remove
import numpy as np
from PIL import Image

# Dossier contenant les vidéos triées par timing approximatif
video_folder = "Videos_gif"
output_folder = "GIFs"
os.makedirs(output_folder, exist_ok=True)  # créer le dossier si nécessaire

# Parcourir tous les fichiers mp4
for filename in os.listdir(video_folder):
    if not filename.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(video_folder, filename)
    # dernière lettre avant ".mp4"
    gif_name = filename[-5] + ".gif"
    gif_path = os.path.join(output_folder, gif_name)

    print(f"Traitement de {filename} -> {gif_name}")

    # Charger la vidéo et sous-clipper selon le timing des vidéos
    clip = VideoFileClip(video_path).subclipped(0.4, 2).resized(width=400)

    frames = []

    for frame in clip.iter_frames(fps=10):  # fps réduit pour GIF léger
        img = Image.fromarray(frame).convert("RGBA")
        img_no_bg = remove(img).convert("RGBA")
        arr = np.array(img_no_bg)
        frames.append(arr)

    # Construire le GIF avec transparence
    gif_clip = ImageSequenceClip(frames, fps=10, with_mask=True)
    gif_clip.write_gif(gif_path)

    print(f"{gif_name} créé")
