**[Lien git du projet](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise)** pour accéder aux scripts détaillés ci dessous

## Détail des scripts :

### Scripts pour l'interface :
- [interface.py](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/interface.py) : Ce script est celui qui nous a permis de créer et de modifier l'interface que vous visualisez actuellement.
- [video_to_gif](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/video_to_gif.py) : Ce script a permis de transformer les vidéos en gif en retirant l'arrière-plan de la vidéo et en mettant le bout de vidéo intéressant au format gif. Il prend en entrée des fichiers mp4 et sort des fichiers gif. Nous utilisons ces gifs dans la page de démonstration pour avoir un exemple pour chaque signe.

### Scripts pour le traitement des données : 
- [trier_corpus](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/trier_corpus.py) : Ce script a permis de récupérer les images et de les ranger dans des sous-dossiers par lettre dans un dossier "corpus_lsf".

- [mediapipe_extraction](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/mediapipe_extraction.py) : Ce script utilise le module MediaPipe et le modèle [HandLandmarker](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/hand_landmarker.task) pour extraire les gestes des vidéos. Il prend en entrée des vidéos (fichiers mp4 ou mov) et sort des fichiers npy et des vidéos (fichiers mp4) pour débuguer si besoin. Le modèle détermine 21 points à différents endroits de la main et les stocke sous forme de vecteurs dans des fichiers npy. Pour être sûres que les mains étaient bien captées et qu'aucun mouvement ne parasitait la compréhension, nous avons fait en sorte que le script enregistre les vidéos affichant les points détectés. Nous avons donc pu nous assurer que le modèle captait bien le mouvement des mains, et que les vecteurs dans les fichiers npy (par la suite transmis à notre modèle) étaient bien des points de la main. AJOUTER UN EXEMPLE DE VIDEO DE DEBUGAGE ?
Le script contient les fonctions suivantes:

    - traiter_image: cette fonction traite l'image, trouve des points sur la main avec mediapipe et crée une image de sortie (utilisée pour débuguer) sur laquelle elle ajoute les points trouvés. Elle stocke ensuite ces points dans un fichier npy. Pour traiter l'image, on utilise le running mode "IMAGE" de HandLandmarker.
    
    - traiter_video: cette fonction lit la vidéo, trouve des points sur la main avec mediapipe et crée une vidéo de sortie (utilisée pour débuguer) sur laquelle elle ajoute les points trouvés. Puis elle stocke les points dans un fichier npy. Pour traiter la vidéo, on utilise le running mode "VIDEO" de HandLandmarker.
    
    - boucle principale: cette boucle parcourt le corpus. Elle applique à chaque fichier la fonction traiter_image s'il a comme extension '.jpg', '.jpeg' ou '.png', et la fonction traiter_video s'il a comme extension '.mp4' ou '.mov'. Si l'extension du fichier trouvé ne figure pas parmi ces extensions, il un message d'erreur est renvoyé.

- [longueur_sequence](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/longueur_sequence.py) : Ce script parcourt les fichiers npy créés par le script mediapipe_extraction et calcule des statistiques sur chaque séquence. Il crée aussi un dataset PyTorch en tronquant/en faisant du padding sur les séquences pour qu'elles aient toutes la même longueur. Il prend en entrée le corpus prétraité (composé de fichiers npy)
Il est séparé en deux parties:

     - Analyse statistique des séquences: le script parcourt le corpus et effectue plusieurs opérations statistiques; il calcule la longueur moyenne des séquences, la longueur maximale, la longueur minimale, et le nombre de fichiers aberrants (pas de main détectée). Il renvoie un graphique des statistiques des longueurs, créé avec matplotlib.
     
     - Création du dataset: cette partie du script initialise la classe LSFDataset avec PyTorch, puis crée le dataset avec nos données en parcourant le corpus.
Les nouveaux fichiers npy paddés sont ensuite ajoutés dans un nouveau dossier (corpus_pretraite_padded), et le dataset dans un fichier dataset_lsf.pt. 

### Script pour l'analyse des données :
- [rnn_models.py](https://github.com/KerenDague/Detection_et_analyse_de_dactylologie_francaise/blob/main/rnn_models.py) : Ce script contient le modèle (réseau de neurones LSTM créé avec torch), ainsi qu'un script de chargement des données, d'entraînement et de test du modèle. Il renvoie les résultats du modèle et une matrice de confusion.
Il est composé de plusieurs parties:

    - Initialisation du modèle LSFTranslator avec PyTorch
    
    - Fonction de chargement des données puis utilisation de cette fonction (load_data) pour charger les données du corpus depuis un dossier
    
    - Séparation du dataset en train et test. On utilise une séparation standard, 80% pour le train et 20% pour le test.
    
    - Fonctions d'entraînement et de test du modèle (train_model, test_model).
    
    - Entraînement et test du modèle puis affichage des résultats et d'une matrice de confusion.
