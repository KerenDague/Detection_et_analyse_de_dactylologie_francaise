## Bibliothèques utilisées :

### Bibliothèques utilisées pour le traitement des données :
- **MediaPipe :** Bibliothèque open source développée par Google, conçue pour appliquer des pipelines de traitement de données multimédia (vidéo, audio, images) en temps réel. Dans ce projet, on utilise son module HandLandmarker qui détecte et localise 21 points clés sur la main (articulations, bout des doigts, paume) à partir d'une image ou d'une vidéo. Ces points, appelés landmarks, sont retournés sous forme de coordonnées normalisées et constituent la base des vecteurs transmis au modèle de classification.

- **OpenCV :** (Open Source Computer Vision Library) Bibliothèque open source spécialisée dans le traitement d'images et de vidéos en temps réel. Dans ce projet, elle est utilisée à deux niveaux : d'une part dans les scripts d'extraction pour lire les fichiers vidéo, dessiner les landmarks détectés par MediaPipe sur les frames et écrire les vidéos de débogage.
