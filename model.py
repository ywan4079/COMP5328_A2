import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from dataset import train_val_split, build_test_loader, ImageDataset
import numpy as np
from utils import Baseline_train, test, train_predict, Hybrid_train

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
            if nal_layer:
                self.nal.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                if nal_layer:
                    clean_probs = F.softmax(outputs, dim=1)
                    # T = F.softmax(self.nal.transition_matrix, dim=1)
                    # noisy_probs = torch.clamp(clean_probs @ T, 1e-12, 1)
                    noisy_probs = self.nal.forward(clean_probs)
                    loss = F.nll_loss(torch.log(noisy_probs), labels)

                    # Test
                    # log_clean = F.log_softmax(outputs, dim=1)
                    # logT = F.log_softmax(self.nal.logits, dim=1) 
                    # log_noisy = torch.logsumexp(log_clean.unsqueeze(2) + logT.unsqueeze(0), dim=1)
                    # loss = F.nll_loss(log_noisy, labels)
                else:
                    train_dataset.transition_matrix = train_dataset.transition_matrix.to(self.device)
                    loss = forward_loss_calculation(outputs, labels, train_dataset.transition_matrix)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            training_loss = running_loss / len(train_loader)

            # Validation phase
            self.model.eval()
            if nal_layer:
                self.nal.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    if nal_layer:
                        clean_probs = F.softmax(outputs, dim=1)
                        # T = F.softmax(self.nal.transition_matrix, dim=1)
                        # noisy_probs = torch.clamp(clean_probs @ T, 1e-12, 1)
                        noisy_probs = self.nal.forward(clean_probs)
                        loss = F.nll_loss(torch.log(noisy_probs), labels)
                        # Test
                        # log_clean = F.log_softmax(outputs, dim=1)
                        # logT = F.log_softmax(self.nal.logits, dim=1) 
                        # log_noisy = torch.logsumexp(log_clean.unsqueeze(2) + logT.unsqueeze(0), dim=1)
                        # loss = F.nll_loss(log_noisy, labels)
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
    
    def predict(self, test_dataset: ImageDataset, nal_layer=False):
        test_dataset.transform = self.model_transform
        test_loader = build_test_loader(test_dataset)

        self.model.eval()
        if nal_layer:
            self.nal.eval()
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
        
# class NoiseAdaptionLayer(nn.Module):
#     def __init__(self, num_classes: int):
#         super().__init__()
#         self.num_classes = num_classes
#         # logits will be updated later

#     def forward(self, clean_prob: torch.Tensor) -> torch.Tensor:
#         T = F.softmax(self.logits, dim=1)
#         noisy_prob = torch.clamp(clean_prob @ T, 1e-9, 1)
#         return noisy_prob

class NoiseLayer(nn.Module):
    def __init__(self, theta, k):
        super(NoiseLayer, self).__init__()
        self.theta = nn.Linear(k, k, bias=False)
        self.theta.weight.data = nn.Parameter(theta)
        self.eye = torch.Tensor(np.eye(k))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        theta = self.eye.to(x.device).detach()
        theta = self.theta(theta)
        theta = self.softmax(theta)
        out = torch.matmul(x, theta)
        return out


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

        self.baseline_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # 11.7M
        self.baseline_model.fc = nn.Linear(self.baseline_model.fc.in_features, num_classes)
        self.baseline_model = self.baseline_model.to(self.device)
        self.model_transform = models.ResNet18_Weights.DEFAULT.transforms()
        self.num_classes = num_classes

        self.optimizer = optimizer(self.baseline_model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # self.nal = NoiseLayer(num_classes).to(self.device)
        # self.optimizer = optimizer(
        #     list(self.model.parameters()) + list(self.nal.parameters()),
        #     lr=self.learning_rate
        # )

    def get_NAL_params(self, basline_model: ModelBase, train_dataset: ImageDataset):
        y_true_noise, y_pred = basline_model.predict(train_dataset)
        baseline_cm = np.zeros((self.nal.num_classes, self.nal.num_classes))
        for n, p in zip(y_true_noise, y_pred):
            baseline_cm[p, n] += 1

        baseline_cm /= baseline_cm.sum(axis=1, keepdims=True)
        baseline_cm = np.log(baseline_cm + 1e-9)

        baseline_cm = nn.Parameter(torch.from_numpy(baseline_cm).to(device=self.device, dtype=torch.float32))
        baseline_cm.requires_grad = True

        return baseline_cm

    def train(self, train_dataset: ImageDataset, val_dataset: ImageDataset):
        train_dataset.transform = self.model_transform
        val_dataset.transform = self.model_transform
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        epoch_no_improvement = 0
        best_model_parameters = None
        for epoch in range(self.num_epochs):
            baseline_train_acc_list, baseline_train_loss_list = Baseline_train(train_loader, self.baseline_model, self.optimizer, self.criterion)
            noise_test_acc_list, noise_test_loss_list = test(val_loader, self.baseline_model, self.criterion)
            avg_val_loss = np.mean(noise_test_loss_list)

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
            self.baseline_model.load_state_dict(best_model_parameters)
        print("Finish training baseline model")
        
        Baseline_output, y_train_noise = train_predict(self.baseline_model, train_loader)
        Baseline_confusion = np.zeros((self.num_classes, self.num_classes))
        for n, p in zip(y_train_noise, Baseline_output):
            n = n.cpu().numpy()
            p = p.cpu().numpy()
            Baseline_confusion[p, n] += 1.
        # noisy channel
        channel_weights = Baseline_confusion.copy()
        channel_weights /= channel_weights.sum(axis=1, keepdims=True)
        channel_weights = np.log(channel_weights + 1e-8)
        channel_weights = torch.from_numpy(channel_weights)  # numpy.ndarray -> tensor
        channel_weights = channel_weights.float()
        noisemodel = NoiseLayer(theta=channel_weights.to(self.device), k=self.num_classes)
        noise_optimizer = torch.optim.Adam(noisemodel.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=1e-4)
        print("noisy channel finished.")
        best_val_loss = float('inf')
        epoch_no_improvement = 0
        best_model_parameters = None
        for epoch in range(self.num_epochs):
            print('Revision Epoch:', epoch)
            noise_train_acc_list, noise_train_loss_list = Hybrid_train(train_loader, self.baseline_model, noisemodel,
                                                                       self.optimizer, noise_optimizer, self.criterion)
            print("After hybrid, test acc: ")
            noise_test_acc_list, noise_test_loss_list = test(val_loader, self.baseline_model, self.criterion)
            avg_val_loss = np.mean(noise_test_loss_list)
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
            self.baseline_model.load_state_dict(best_model_parameters)
        print("Finished hybrid training.")



        # noisy_cnn = CNN(num_classes=self.model.fc.out_features, dataset_name=self.dataset_name, num_epochs=self.num_epochs, learning_rate=self.learning_rate, batch_size=self.batch_size, patience=self.patience, criterion=self.criterion)
        # train_dataset.transition_matrix = torch.eye(self.model.fc.out_features).to(self.device) # assume not noisy
        # print(f"Training noisy CNN to estimate NAL Layer params...")
        # noisy_cnn.train(train_dataset, val_dataset)
        # self.nal.logits = self.get_NAL_params(noisy_cnn, train_dataset)
        # self.nal.logits = self.nal.logits.to(self.device)

        # super().train(train_dataset, val_dataset, nal_layer=True)

    def predict(self, test_dataset: ImageDataset):       
        super().predict(test_dataset)
    


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