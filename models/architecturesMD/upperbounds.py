import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)


def start_upperbound(models, model_names, device):
  test_loader = load_dataloader([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

  for i in range(len(models)):
    print("Computing upperbound for model: " + model_names[i])
    compute_upperbound(models[i], model_names[i], test_loader, device)

def load_dataloader(target_labels):
    batch_size = 64

    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                      ])

    test_dataset = torchvision.datasets.CIFAR100(root='./data_test', train=True, download=True, transform=transform)

    filtered_indices = [idx for idx, label in enumerate(test_dataset.targets) if label in target_labels]
    filtered_dataset = torch.utils.data.Subset(test_dataset, filtered_indices)

    test_loader = torch.utils.data.DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

    return test_loader

def compute_upperbound(model, model_name, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0004)
    early_stopping = EarlyStopping(patience=10, delta=0.001, path=f'best_{model_name}.pth')
    
    model.train()
    correct = 0
    total = 0

    for epoch in range(75):
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy at epoch {epoch+1}: {accuracy:.2f}%")

        # Validation step
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for val_images, val_labels in test_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_loss /= len(test_loader)
        val_accuracy = 100 * val_correct / val_total
        print(f"Validation accuracy at epoch {epoch+1}: {val_accuracy:.2f}%")

        # Check for early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

