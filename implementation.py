import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class SMEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SMEClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train_model(model, criterion, optimizer, train_loader, epochs=100):
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

if __name__ == '__main__':
    # Dummy data generation
    np.random.seed(42)
    num_samples = 500
    input_dim = 10
    num_classes = 5

    # Generate random features and labels
    X = np.random.rand(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # Split into train and test sets
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_X, test_X = torch.split(X_tensor, [train_size, test_size])
    train_y, test_y = torch.split(y_tensor, [train_size, test_size])

    # Create DataLoader
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model parameters
    hidden_dim = 64
    model = SMEClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, epochs=50)

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")