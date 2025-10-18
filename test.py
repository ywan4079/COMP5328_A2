from dataset import ImageDataset, load_dataset
from model import CNN, VisionTransformer
import torch
from sklearn.metrics import accuracy_score

torch.cuda.empty_cache()  # Clear GPU memory

training_data, training_labels, testing_data, testing_labels = load_dataset('datasets/FashionMNIST0.3.npz')
T = torch.tensor([[0.7, 0.3, 0.0],
                  [0.0, 0.7, 0.3],
                  [0.3, 0.0, 0.7]], dtype=torch.float32)
train_dataset = ImageDataset(training_data, training_labels, transition_matrix=T)
test_dataset = ImageDataset(testing_data, testing_labels)

cnn = CNN(num_classes=3)
cnn.train(train_dataset)
y_true, y_pred = cnn.predict(test_dataset)
print(f"CNN Test Acc: {accuracy_score(y_true, y_pred)*100:.2f}%")

# vit = VisionTransformer(num_classes=3, transition_matrix=T, batch_size=32)
# vit.train(train_dataset)
# y_true, y_pred = vit.predict(test_dataset)
# print(f"ViT Test Acc: {accuracy_score(y_true, y_pred)*100:.2f}%")
