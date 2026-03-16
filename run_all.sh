#!/bin/bash

# Activer l'environnement virtuel
source /chemin

# Lance les scripts Python 
python trier_corpus.py
wait
python mediapipe_extraction.py
wait
python longueur_sequence.py
wait
python rnn_models.py
wait

# Désactiver l'environnement à la fin
deactivate
