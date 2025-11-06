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

def add_noise(img):
    noise = np.random.normal(0, 10, img.shape)  # mean=0, std=10
    img = np.clip(img + noise, 0, 255)
    return img

def flip(img):
    return np.fliplr(img)

def zoom_in(img, zoom_factor=0.8):
    h, w = img.shape[:2]
    zh, zw = int(h * zoom_factor), int(w * zoom_factor)
    top, left = (h - zh) // 2, (w - zw) // 2 # in the middle
    zoomed_img = img[top:top+zh, left:left+zw]
    zoomed_img = np.array(Image.fromarray(zoomed_img).resize((w, h)))
    return zoomed_img


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

def data_augmentation(X: np.array, Y: np.array):
    X_processed = []
    Y_processed = []
    for idx, img in enumerate(X):
        img = np.array(img)
        X_processed.append(img)
        noisy_img = add_noise(img)
        X_processed.append(noisy_img)
        flipped_img = flip(img)
        X_processed.append(flipped_img)
        zoomed_img = zoom_in(img)
        X_processed.append(zoomed_img)

        Y_processed.extend([Y[idx]]*4)

    return np.array(X_processed, dtype=np.uint8), np.array(Y_processed)

# def data_augmentation(dataset: ImageDataset, dataset_transform):
#     X_processed = []
#     Y_processed = []
#     for img, label in dataset.da:
#         img = np.array(img)
#         X_processed.append(img)
#         noisy_img = add_noise(img)
#         X_processed.append(noisy_img)
#         flipped_img = flip(img)
#         X_processed.append(flipped_img)
#         zoomed_img = zoom_in(img)
#         X_processed.append(zoomed_img)

#         Y_processed.extend([label]*4)

#     return ImageDataset(X_processed, Y_processed, transform=dataset_transform, transition_matrix=dataset.transition_matrix)


# def train_val_split(training_data: ImageDataset, ratio: float = 0.2, batch_size: int = 128):
#     train_idx, val_idx = train_test_split(list(range(len(training_data))), test_size=ratio)
#     train_subset = Subset(training_data, train_idx)
#     print("Before: ", len(train_subset))
#     train_subset = data_augmentation(train_subset, training_data.transform)
#     print("After: ", len(train_subset))
#     val_subset = Subset(training_data, val_idx)

#     train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
#     return train_loader, val_loader

def train_val_split(training_data: np.array, training_labels: np.array, ratio: float = 0.2):
    train_idx, val_idx = train_test_split(list(range(len(training_data))), test_size=ratio)
    training_subset = training_data[train_idx]
    validation_subset = training_data[val_idx]
    training_sub_labels = training_labels[train_idx]
    validation_sub_labels = training_labels[val_idx]
    return training_subset, training_sub_labels, validation_subset, validation_sub_labels

def build_test_loader(test_data: ImageDataset, batch_size: int = 128):
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader


if __name__ == "__main__":
    training_data, training_labels, testing_data, testing_labels = load_dataset('datasets/FashionMNIST0.6.npz')
    train_dataset = ImageDataset(training_data, training_labels)

    plt.imshow(train_dataset[0][0], cmap='gray')
    plt.show()