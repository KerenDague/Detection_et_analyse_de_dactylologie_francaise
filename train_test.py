from make_dataset import LSFDataset
from torch.utils.data import DataLoader, random_split

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
base_path = 'corpus_pretraite'

lsf_dataset = LSFDataset(base_path, letters)

# Séparation en train/test
train_size = int(0.8 * len(lsf_dataset))
test_size = len(lsf_dataset) - train_size
train_dataset, test_dataset = random_split(lsf_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)