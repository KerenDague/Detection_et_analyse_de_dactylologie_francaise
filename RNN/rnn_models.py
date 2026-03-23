import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

""" 
Création d'un réseau de neurones qui prend les fichiers npy en entrée.
Les fichiers npy sont constitués de "groupes" de 21 points (1 groupe par image) qui ont chacun 3 coordonnées (d'où le `input_size = 63`)
"""

# Le modèle LSTM
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

# Fonction pour charger les données depuis un dossier
def load_data(base_path, letters):
    data = []
    labels = []
    label_map = {letter: idx for idx, letter in enumerate(letters)}

    for letter in letters:
        letter_path = os.path.join(base_path, letter)
        if not os.path.exists(letter_path):
            continue
        for file in os.listdir(letter_path):
            if file.endswith('.npy'):
                file_path = os.path.join(letter_path, file)
                sequence = np.load(file_path)
                data.append(sequence)
                labels.append(label_map[letter])

    return np.array(data), np.array(labels)

# Chemin et lettres
base_path = 'corpus_pretraite_padded'
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#Chargement des données
X, y = load_data(base_path, letters)

# Conversion en tenseurs PyTorch
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Création du TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)

# Séparation en train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Création des DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialisation du modèle, fonction de perte et de l'optimiseur
model = LSFTranslator()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Entrainement
def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return train_losses, train_accuracies

# Test
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    return y_true, y_pred

#Matrice de confusion
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Entrainement du modèle
num_epochs = 25
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluation du modèle
y_true, y_pred = test_model(model, test_loader, criterion)

#Matrice de confusion
plot_confusion_matrix(y_true, y_pred, letters)

#courbes d'apprentissage
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()
