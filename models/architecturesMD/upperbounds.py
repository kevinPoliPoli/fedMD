import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from utils.custom_dataloader import CustomDataset

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


def start_upperbound(train_data, test_data, models, model_names, device):
  trainset = CustomDataset(train_data)
  testset = CustomDataset(test_data)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
  testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

  upperbounds = []
  for i in range(len(models)):
    print("Computing upperbound for model: " + model_names[i])
    upperbounds.append(compute_upperbound(trainloader, testloader, models[i], model_names[i], device))
  
  return upperbounds


def compute_upperbound(trainloader, testloader, model, model_name, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0004)
    early_stopping = EarlyStopping(patience=10, delta=0.001, path=f'best_{model_name}.pth')
    
    model.train()
    correct = 0
    total = 0
    val_accuracy = 0
    for epoch in range(75):
        for images, labels in trainloader:
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
            for val_images, val_labels in testloader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_loss /= len(testloader)
        val_accuracy = 100 * val_correct / val_total
        print(f"Validation accuracy at epoch {epoch+1}: {val_accuracy:.2f}%")

        # Check for early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            return val_accuracy
            break
            
    return val_accuracy
        
import matplotlib.pyplot as plt

def plot_accuracy_epochs(epochs, accuracies_list, upper_bounds):
    for i, accuracies in enumerate(accuracies_list):
        color = plt.cm.get_cmap('tab10')(i)
        plt.plot(range(1, epochs + 1), accuracies, label=f"Model {i + 1}", color=color)

    for i, upper_bound in enumerate(upper_bounds):
        color = plt.cm.get_cmap('tab10')(i)
        plt.axhline(y=upper_bound, linestyle='--', label=f"Model {i + 1} Upper Bound", color=color, xmin=3/epochs, xmax=1)


    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('CIFAR100 I.I.D.')

    plt.xticks(range(1, epochs + 1))  

    plt.legend()
    plt.show()


