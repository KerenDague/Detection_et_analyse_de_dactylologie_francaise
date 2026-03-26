import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

""" 
Création d'un réseau de neurones qui prend les fichiers npy en entrée.
Les fichiers npy sont constitués de "groupes" de 21 points (1 groupe par image) qui ont chacun 3 coordonnées (d'où le `input_size = 63`)
"""

MAX_FRAMES  = 80 
INPUT_SIZE  = 63
HIDDEN_SIZE = 256
NUM_CLASSES = 26
AUGMENT_SUFFIXES = ('_miroir', '_luminosite', '_rotation')


# Modèle LSTM bidirectionnel

class LSFTranslator(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
        )
        self.dropout    = nn.Dropout(0.4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(self.dropout(out.mean(dim=1)))


# Chargement des données 

def is_augmented(filename):
    """Retourne True si le fichier est une augmentation (pas un original)."""
    stem = os.path.splitext(filename)[0]
    return any(stem.endswith(s) for s in AUGMENT_SUFFIXES)


def get_original_stem(filename):
    """Retourne le nom de la vidéo originale à partir d'un fichier augmenté."""
    stem = os.path.splitext(filename)[0]
    for suffix in AUGMENT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_data(base_path, letters, test_ratio=0.2, max_frames=MAX_FRAMES, input_size=INPUT_SIZE):
    """
    Charge les données depuis corpus_pretraite_padded/lettre/fichier.npy.
    
    La séparation train/test se fait par vidéo originale : si une originale
    part dans le test, toutes ses augmentations partent aussi dans le test,
    ce qui évite le data leakage.
    """
    label_map = {letter: idx for idx, letter in enumerate(letters)}
    train_data, train_labels = [], []
    test_data,  test_labels  = [], []
    skipped = 0

    for letter in letters:
        letter_path = os.path.join(base_path, letter)
        if not os.path.exists(letter_path):
            print(f"Dossier manquant : {letter_path}")
            continue

        all_files = [f for f in os.listdir(letter_path) if f.endswith('.npy')]

        # Récupérer les stems des vidéos originales uniquement
        original_stems = list({
            get_original_stem(f) for f in all_files if not is_augmented(f)
        })

        # Tirage aléatoire des originaux qui vont dans le test
        n_test = max(1, int(len(original_stems) * test_ratio))
        test_stems = set(original_stems[:n_test])   # on prend les n_test premiers (ordre fixe)

        for f in all_files:
            path = os.path.join(letter_path, f)
            seq  = np.load(path, allow_pickle=True)

            # Ignorer les séquences invalides (déjà filtrées dans reg_seq.py, sécurité en plus)
            if seq.ndim != 2 or seq.shape[1] != input_size or seq.shape[0] == 0:
                skipped += 1
                continue

            # Les séquences sont déjà paddées/tronquées à MAX_FRAMES par reg_seq.py
            # On re-vérifie quand même la shape pour être sûr
            if seq.shape[0] != max_frames:
                # tronquer ou padder si nécessaire
                if seq.shape[0] > max_frames:
                    seq = seq[:max_frames]
                else:
                    pad = np.zeros((max_frames - seq.shape[0], input_size), dtype=np.float32)
                    seq = np.vstack([seq, pad])

            seq = seq.astype(np.float32)

            # Un fichier (original ou augmenté) va dans le test si son original est dans test_stems
            if get_original_stem(f) in test_stems:
                test_data.append(seq)
                test_labels.append(label_map[letter])
            else:
                train_data.append(seq)
                train_labels.append(label_map[letter])

    print(f"Train : {len(train_data)} séquences  |  Test : {len(test_data)} séquences  |  Ignorées : {skipped}")
    return (
        np.array(train_data,  dtype=np.float32),
        np.array(train_labels, dtype=np.int64),
        np.array(test_data,   dtype=np.float32),
        np.array(test_labels,  dtype=np.int64),
    )


# Entrainement

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=50):
    train_losses, train_accuracies = [], []
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs.data, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc  = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        scheduler.step(epoch_loss)

        print(f'Epoch {epoch+1:>2}/{num_epochs} — '
              f'Loss: {epoch_loss:.4f}, '
              f'Accuracy: {epoch_acc:.2f}%, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_model.pt')

    print(f"\nMeilleur modèle sauvegardé (loss: {best_loss:.4f}) → best_model.pt")
    return train_losses, train_accuracies


# Evaluation

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            test_loss   += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total       += labels.size(0)
            correct     += (predicted == labels).sum().item()
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    test_loss /= len(test_loader)
    test_acc   = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    return y_true, y_pred


#Matrice de confusion

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matrice de confusion')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.show()


# Main

base_path = 'corpus_pretraite_padded'
letters   = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M','N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Chargement avec séparation train/test sans data leakage
X_train, y_train, X_test, y_test = load_data(base_path, letters, test_ratio=0.2)
print(f"Shape X_train : {X_train.shape}  |  Shape X_test : {X_test.shape}")

# Conversion en tenseurs
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.LongTensor(y_test)

# Normalisation calculée sur le train uniquement (pas de fuite vers le test)
X_mean    = X_train_t.mean(dim=(0, 1), keepdim=True)
X_std     = X_train_t.std(dim=(0, 1),  keepdim=True) + 1e-8
X_train_t = (X_train_t - X_mean) / X_std
X_test_t  = (X_test_t  - X_mean) / X_std

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=32, shuffle=False)

# Modèle
model     = LSFTranslator()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Entrainement
num_epochs = 50
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)

# Chargement du meilleur modèle
model.load_state_dict(torch.load('best_model.pt'))
print("Meilleur modèle chargé pour l'évaluation.")

# Evaluation
y_true, y_pred = test_model(model, test_loader, criterion)

# Sauvegarde complète pour l'interface web
torch.save({
    'model_state_dict': model.state_dict(),
    'X_mean':  X_mean,
    'X_std':   X_std,
    'letters': letters,
    'max_frames': MAX_FRAMES,
}, 'lsf_model.pt')
print("Modèle complet sauvegardé → lsf_model.pt")

# Matrice de confusion
plot_confusion_matrix(y_true, y_pred, letters)

# Courbes d'apprentissage
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, color='#e075a0', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, color='#8B2240', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
