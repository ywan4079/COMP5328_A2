import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from dataset import train_val_split, build_test_loader, ImageDataset
import numpy as np

def forward_loss_calculation(outputs, noisy_labels, transition_matrix): # this is cross-entropy loss with loss correction
    if transition_matrix is None:
        return nn.CrossEntropyLoss()(outputs, noisy_labels)
    prob_clean = F.softmax(outputs, dim=1)
    prob_noisy = prob_clean @ transition_matrix
    log_prob_noisy = torch.log(prob_noisy + 1e-12)  # Adding a small constant for numerical stability
    loss = F.nll_loss(log_prob_noisy, noisy_labels)
    return loss

def all_softmax(model, data_loader, device):
    model.eval()
    all_probs = [] # probs N x C
    for images, _ in data_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    return all_probs

def estimate_T_anchor(probs, min_conf = 0.8):
    # probs: N x C, pred_labels: N
    N, C = probs.shape
    T_hat = torch.zeros((C, C), dtype=torch.float32)

    for i in range(C):
        probs_i = probs[:, i] # N
        sorted_probs_idx = torch.argsort(probs_i, descending=True)
        candidates_idx = sorted_probs_idx[probs_i[sorted_probs_idx] >= min_conf]
        if len(candidates_idx) == 0:
            candidates_idx = sorted_probs_idx[:int(0.1 * N)]  # take top 10% if no candidates above min_conf
        
        T_hat[i] = probs[candidates_idx].mean(dim=0)

    return T_hat

class ModelBase:
    def __init__(self, num_epochs: int = 100, dataset_name: str = "", learning_rate: float = 0.001, batch_size: int = 128, patience: int = 10, criterion=nn.CrossEntropyLoss()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.criterion = criterion
        self.model = None
        self.model_transform = None

    def train(self, train_dataset: ImageDataset, val_dataset: ImageDataset, nal_layer: bool = False):
        train_dataset.transform = self.model_transform
        val_dataset.transform = self.model_transform
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        # train_loader, val_loader = train_val_split(train_dataset, batch_size=self.batch_size)

        best_val_loss = float('inf')
        epoch_no_improvement = 0
        best_model_parameters = None

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                if nal_layer:
                    clean_probs = F.softmax(outputs, dim=1)
                    noisy_probs = self.nal(clean_probs)
                    loss = F.nll_loss(torch.log(noisy_probs), labels)
                else:
                    train_dataset.transition_matrix = train_dataset.transition_matrix.to(self.device)
                    loss = forward_loss_calculation(outputs, labels, train_dataset.transition_matrix)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            training_loss = running_loss / len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    if nal_layer:
                        clean_probs = F.softmax(outputs, dim=1)
                        noisy_probs = self.nal(clean_probs)
                        loss = F.nll_loss(torch.log(noisy_probs), labels)
                    else:
                        loss = forward_loss_calculation(outputs, labels, train_dataset.transition_matrix)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss/len(val_loader)
            val_accuracy = 100 * correct / total

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {training_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epoch_no_improvement = 0
                best_model_parameters = self.model.state_dict()
            else:
                epoch_no_improvement += 1
                if epoch_no_improvement >= self.patience:
                    print(f"No improvement for {self.patience} epochs. Early stopping.")
                    break
        
        if best_model_parameters:
            self.model.load_state_dict(best_model_parameters)
    
    def predict(self, test_dataset: ImageDataset):
        test_dataset.transform = self.model_transform
        test_loader = build_test_loader(test_dataset)

        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        return y_true, y_pred
        
class NoiseAdaptionLayer(nn.Module):
    def __init__(self, num_classes: int, transition_matrix: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        else:
            epsilon = 1e-9
            t = torch.full((num_classes, num_classes), fill_value=epsilon / (num_classes - 1))
            for i in range(num_classes):
                t[i, i] = 1.0 - epsilon
            t = torch.log(t)
            self.transition_matrix = nn.Parameter(t)
        self.transition_matrix.requires_grad = True


    def forward(self, clean_prob: torch.Tensor) -> torch.Tensor:
        T = F.softmax(self.transition_matrix, dim=1)
        noisy_prob = torch.clamp(clean_prob @ T, 1e-12, 1)
        return noisy_prob


class CNN(ModelBase):
    def __init__(self, num_classes: int, dataset_name: str = "", num_epochs: int = 100, learning_rate: float = 0.001, batch_size: int = 128, patience: int = 10, criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam):
        super().__init__(num_epochs, dataset_name, learning_rate, batch_size, patience, criterion)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # 11.7M
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        self.model_transform = models.ResNet18_Weights.DEFAULT.transforms()
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_dataset: ImageDataset, val_dataset: ImageDataset):
        if train_dataset.transition_matrix is None:
            noisy_cnn = CNN(num_classes=self.model.fc.out_features, dataset_name=self.dataset_name, num_epochs=self.num_epochs, learning_rate=self.learning_rate, batch_size=self.batch_size, patience=self.patience, criterion=self.criterion)
            train_dataset.transition_matrix = torch.eye(self.model.fc.out_features).to(self.device) # assume not noisy
            print(f"Training noisy CNN to estimate transition matrix...")
            noisy_cnn.train(train_dataset, val_dataset)
            probs = all_softmax(noisy_cnn.model, DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False), self.device)
            T_hat = estimate_T_anchor(probs)
            train_dataset.transition_matrix = T_hat.T
            val_dataset.transition_matrix = T_hat.T
        super().train(train_dataset, val_dataset)
        # torch.save(self.model.state_dict(), f'models/resnet18_{self.dataset_name}_model_{round}.pth')


class CNNWithNAL(ModelBase):
    def __init__(self, num_classes: int, dataset_name: str = "", num_epochs: int = 100, learning_rate: float = 0.001, batch_size: int = 128, patience: int = 10, criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam):
        super().__init__(num_epochs, dataset_name, learning_rate, batch_size, patience, criterion)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # 11.7M
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        self.model_transform = models.ResNet18_Weights.DEFAULT.transforms()
        self.nal = NoiseAdaptionLayer(num_classes).to(self.device)
        self.optimizer = optimizer(
            list(self.model.parameters()) + list(self.nal.parameters()),
            lr=self.learning_rate
        )

    def train(self, train_dataset: ImageDataset, val_dataset: ImageDataset):
        super().train(train_dataset, val_dataset, nal_layer=True)
    


class VisionTransformer(ModelBase):
    def __init__(self, num_classes: int, dataset_name: str = "", num_epochs: int = 100, learning_rate: float = 0.001, batch_size: int = 128, patience: int = 10, criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam):
        super().__init__(num_epochs, dataset_name, learning_rate, batch_size, patience, criterion)
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1) # 86M
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        self.model = self.model.to(self.device)
        self.model_transform = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_dataset: ImageDataset, round: int):
        super().train(train_dataset)
        torch.save(self.model.state_dict(), f'models/vitb16_{self.dataset_name}_model_{round}.pth')