import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from dataset import train_val_split, build_test_loader, ImageDataset

def loss_calculation(outputs, noisy_labels, transition_matrix): # this is cross-entropy loss with loss correction
    if transition_matrix is None:
        return nn.CrossEntropyLoss()(outputs, noisy_labels)
    prob_clean = F.softmax(outputs, dim=1)
    prob_noisy = (transition_matrix @ prob_clean.T).T
    log_prob_noisy = torch.log(prob_noisy + 1e-12)  # Adding a small constant for numerical stability
    loss = F.nll_loss(log_prob_noisy, noisy_labels)
    return loss

class ModelBase:
    def __init__(self, num_epochs: int = 100, learning_rate: float = 0.001, batch_size: int = 128, patience: int = 10, criterion=nn.CrossEntropyLoss()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.criterion = criterion
        self.model = None
        self.model_transform = None

    def train(self, train_dataset: ImageDataset):
        train_dataset.transform = self.model_transform
        train_loader, val_loader = train_val_split(train_dataset, batch_size=self.batch_size)

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

                loss = loss_calculation(outputs, labels, train_dataset.transition_matrix)
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
                    loss = loss_calculation(outputs, labels, train_dataset.transition_matrix)
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
        

class CNN(ModelBase):
    def __init__(self, num_classes: int, num_epochs: int = 100, learning_rate: float = 0.001, batch_size: int = 128, patience: int = 10, criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam):
        super().__init__(num_epochs, learning_rate, batch_size, patience, criterion)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # 11.7M
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        self.model_transform = models.ResNet18_Weights.DEFAULT.transforms()
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_dataset: ImageDataset):
        super().train(train_dataset)
        torch.save(self.model.state_dict(), 'models/resnet18_model.pth')
    

class VisionTransformer(ModelBase):
    def __init__(self, num_classes: int, num_epochs: int = 100, learning_rate: float = 0.001, batch_size: int = 128, patience: int = 10, criterion=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam):
        super().__init__(num_epochs, learning_rate, batch_size, patience, criterion)
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1) # 86M
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        self.model = self.model.to(self.device)
        self.model_transform = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_dataset: ImageDataset):
        super().train(train_dataset)
        torch.save(self.model.state_dict(), 'models/vitb16_model.pth')