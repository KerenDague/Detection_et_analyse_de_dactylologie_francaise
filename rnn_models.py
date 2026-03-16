import torch.nn as nn
import torch.nn.functional as F
import numpy as np

""" Création d'un réseau de neurones qui prend les fichiers npy en entrée
Les fichiers npy sont constitués de "groupes" de 21 points (1 groupe par image) qui ont chacun 3 coordonnées (d'où le `input_size = 63`)"""

class LSFTranslator(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=26):
        super(LSFTranslator, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_frame = out[:, -1, :]

        prediction = self.classifier(last_frame)

        return prediction