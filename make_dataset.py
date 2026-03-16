import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from longueur_sequence import longueur_max

# Création du dataset
class LSFDataset(Dataset):
    def __init__(self, base_path, letters):
        self.samples = []
        self.letter_map = {letter: num for num, letter in enumerate(letters)}

        for letter in letters:
            npy_path = os.path.join(base_path, letter)

            for file in os.listdir(npy_path):
                if file.endswith('.npy'):
                    self.samples.append((os.path.join(npy_path, file), self.letter_map[letter]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)

        if len(data) > longueur_max:
            data = data[:longueur_max]

        elif len(data) < longueur_max:
            pad_width = longueur_max - len(data)
            padding = np.zeros((pad_width, 63))
            data = np.vstack((data, padding))

        return torch.from_numpy(data).float(), torch.tensor(label)
