from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_dataset(path: str):
    dataset = np.load(path)
    training_data = dataset['Xtr']
    training_labels = dataset['Str']
    testing_data = dataset['Xts']
    testing_labels = dataset['Yts']
    if path.startswith("datasets/FashionMNIST"):
        training_data = training_data.reshape((-1, 28, 28))
        testing_data = testing_data.reshape((-1, 28, 28))
    # else:
    #     training_data = training_data.reshape((-1, 32, 32, 3))
    #     testing_data = testing_data.reshape((-1, 32, 32, 3))
    return training_data, training_labels, testing_data, testing_labels

def scaling(dataset: np.ndarray):
    return dataset.astype(np.float32) / 255.0

class ImageDataset(Dataset):
    def __init__(self, X, Y, transform=None, transition_matrix=None):
        self.images = X
        self.labels = Y
        self.transform = transform
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition_matrix = transition_matrix.to(device) if transition_matrix is not None else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx]).convert('RGB')  # Convert to PIL Image for compatibility
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def train_val_split(training_data: ImageDataset, ratio: float = 0.2, batch_size: int = 128):
    train_idx, val_idx = train_test_split(list(range(len(training_data))), test_size=ratio)
    train_subset = Subset(training_data, train_idx)
    val_subset = Subset(training_data, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def build_test_loader(test_data: ImageDataset, batch_size: int = 128):
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader


if __name__ == "__main__":
    training_data, training_labels, testing_data, testing_labels = load_dataset('datasets/FashionMNIST0.6.npz')
    train_dataset = ImageDataset(training_data, training_labels)

    plt.imshow(train_dataset[0][0], cmap='gray')
    plt.show()