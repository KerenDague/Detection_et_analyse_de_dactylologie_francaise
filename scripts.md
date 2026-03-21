**[Lien git du projet](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise)** pour accéder aux scripts détaillés ci dessous

## Détail des scripts :

### Scripts pour l'interface :
- [interface.py](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/interface.py) : Ce script est celui qui nous a permis de créer et de modifier l'interface que vous visualisez actuellement.
- [video_to_gif](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/video_to_gif.py) : Ce script a permis de transformer les vidéos en gif en retirant le background de la vidéo et en mettant le bout de vidéo intéressant au format gif.

### Scripts pour le traitement des données : 
- [trier_corpus](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/trier_corpus.py) : Ce script a permis de récupérer les images et de les ranger dans des sous-dossiers par lettre dans un dossier "corpus_lsf"
- [mediapipe_extraction](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/mediapipe_extraction.py) : Ce script utilise le module MediaPipe et le modèle [HandLandmarker](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/hand_landmarker.task) pour extraire les gestes des vidéos. Le modèle détermine 21 points à différents endroits de la main et les stocke sous forme de vecteurs dans des fichiers npy. Pour être sûres que les mains étaient bien captées et qu'aucun mouvement ne parasitait la compréhension, nous avons fait en sorte que le script enregistre les vidéos affichant les points détectés. Nous avons donc pu nous assurer que le modèle captait bien le mouvement des mains, et que les vecteurs dans les fichiers npy (par la suite transmis à notre modèle) étaient bien des points de la main. AJOUTER UN EXEMPLE DE VIDEO DE DEBUGAGE ? 
- [longueur_sequence](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/longueur_sequence.py) : Ce script parcourt les fichiers npy créés par le script mediapipe_extraction et calcule des statistiques sur chaque séquence (nombre total de vidéos, longueur moyenne des séquences, longueurs minimale et maximale, fichiers aberrants). Il crée aussi un dataset PyTorch en tronquant/en faisant du padding sur les séquences pour qu'elles aient toutes la même longueur. Les nouveaux fichiers npy paddés sont ajoutés dans un nouveau dossier (corpus_pretraite_padded), et le dataset dans un fichier dataset_lsf.pt.

### Script pour l'analyse des données :
- [rnn_models.py](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/rnn_models.py) : Ce script contient le modèle (réseau de neurones LSTM créé avec torch), ainsi qu'un script de chargement des données, d'entraînement et de test du modèle. Il renvoie les résultats du modèle et une matrice de confusion.
