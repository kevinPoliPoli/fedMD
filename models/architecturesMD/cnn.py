import torch
import torch.nn as nn
import os
import errno
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNN2LayerFCModel(nn.Module):
    def __init__(self, name, n_classes, n1=128, n2=256, dropout_rate=0.2, input_shape=(28, 28)):
        super(CNN2LayerFCModel, self).__init__()

        self.name = name
        self.conv1 = nn.Conv2d(3, n1, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(n1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=3, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(n2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n2 * 16 * 16, n_classes)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.batchnorm1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.batchnorm2(y)
        y = self.relu2(y)
        y = self.dropout2(y)
        y = self.flatten(y)
        y = self.fc(y)
        #y = self.softmax(y)
        return y



class CNN3LayerFCModel(nn.Module):
    def __init__(self, name, n_classes, n1=128, n2=192, n3=256, dropout_rate=0.2, input_shape=(28, 28)):
        super(CNN3LayerFCModel, self).__init__()


        self.name = name
        self.conv1 = nn.Conv2d(3, n1, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(n1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=2, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(n2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(n2, n3, kernel_size=3, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(n3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout2d(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n3 * 3 * 3, n_classes)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.batchnorm1(y)
        y = self.relu1(y)
        y = self.dropout1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.batchnorm2(y)
        y = self.relu2(y)
        y = self.dropout2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.batchnorm3(y)
        y = self.relu3(y)
        y = self.dropout3(y)
        y = self.flatten(y)
        y = self.fc(y)
        #y = self.softmax(y)
        return y


def _returnModel(name, n_classes=16, input_shape=(32,32,3), **model_params):
  if name == "2_layer_CNN":
    model = CNN2LayerFCModel(name, n_classes, input_shape=(32,32,3), **model_params)
    return model
  elif name == "3_layer_CNN":
    model = CNN3LayerFCModel(name, n_classes, input_shape=(32,32,3), **model_params)
    return model


def train_models(models, X_train, y_train, X_test, y_test, device,
                 save_dir="./", save_names=None,
                 early_stopping=True, min_delta=0.001, patience=3,
                 batch_size=128, epochs=20, is_shuffle=True, verbose=1):
    
    resulting_val_acc = []
    record_result = []
    device = device

    X_train = torch.Tensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_shuffle)

    for n, model in enumerate(models):
        print("Training model ", n)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        if early_stopping:
            best_val_acc = 0.0
            patience_counter = 0

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                train_correct = 0

                for inputs, targets in train_loader:
                    inputs = inputs
                    targets = targets

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == targets).sum().item()

                train_loss /= len(X_train)
                train_acc = train_correct / len(X_train)
                
                model.eval()
                with torch.no_grad():
                  val_outputs = model(X_test.to(device))
                  _, val_predicted = torch.max(val_outputs.data, 1)
                  val_acc = (val_predicted == y_test.to(device)).sum().item() / len(X_test)

                if val_acc > best_val_acc + min_delta:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break

                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        else:
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                train_correct = 0

                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == targets).sum().item()

                train_loss /= len(X_train)
                train_acc = train_correct / len(X_train)

                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        resulting_val_acc.append(val_acc)
        record_result.append({"train_acc": train_acc, "val_acc": val_acc, "train_loss": train_loss})

        if save_dir is not None:
            save_dir_path = os.path.abspath(save_dir)
            # make dir
            try:
                os.makedirs(save_dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            if save_names is None:
              file_name = os.path.join(save_dir, "model_{0}.pth".format(n))
            else:
              file_name = os.path.join(save_dir, save_names[n] + ".pth")
            
            torch.save(model.state_dict(), file_name)

        torch.cuda.empty_cache()  # Free up GPU memory

    print("pre-train accuracy: ")
    print(resulting_val_acc)

    return record_result