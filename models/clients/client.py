import copy
import numpy as np
import random
import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from baseline_constants import ACCURACY_KEY
from datetime import datetime
import importlib
import os

from utils.custom_dataloader import CustomDataset, ConsensusDataset

class Client:

    def __init__(self, seed, client_id, lr, weight_decay, batch_size, momentum, train_data, eval_data, private_test, model, device=None,
                 num_workers=0, run=None, mixup=False, mixup_alpha=1.0):

        self._model = model
        self.id = client_id

        self.train_data = CustomDataset(train_data)
        self.eval_data = CustomDataset(eval_data)
        self.testloader = torch.utils.data.DataLoader(self.eval_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        self.private_test_X = torch.Tensor(private_test['X'])
        self.private_test_y = torch.Tensor(private_test['y'])
        
        self.seed = seed
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.run = run
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha # α controls the strength of interpolation between feature-target pairs
    

    def transferLearningInit(self, models_parameters, num_epochs=25, batch_size=32, min_delta=0.001, patience=10, verbose=1): 
      print(f"Training private model: {self.model.name} of client: {self.id}")
      dl = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)

      criterion = nn.CrossEntropyLoss().to(self.device)

      print(f'lr of {self.model.name}:{models_parameters[self.model.name]["lr"]}')
      print(f'wd of {self.model.name}:{models_parameters[self.model.name]["weight_decay"]}')

      wd = models_parameters[self.model.name]['weight_decay']
      if wd != None:
        optimizer = optim.Adam(self.model.parameters(), lr=models_parameters[self.model.name]['lr'], weight_decay = models_parameters[self.model.name]['weight_decay'])
      else:
        optimizer = optim.Adam(self.model.parameters(), lr=models_parameters[self.model.name]['lr'])

      best_val_acc = 0.0
      patience_counter = 0
      should_load = False

      for epoch in range(num_epochs):
          self.model.train()
          train_loss = 0.0
          train_correct = 0

          for inputs, targets in dl:
              inputs = inputs.to(self.device)
              targets = targets.to(self.device)

              optimizer.zero_grad()
              outputs = self.model(inputs)
              loss = criterion(outputs, targets)
              loss.backward()
              optimizer.step()

              train_loss += loss.item() * inputs.size(0)

              inputs = inputs.cpu()
              targets = targets.cpu()
              _, predicted = torch.max(outputs.data, 1)
              train_correct += (predicted.cpu() == targets).sum().item()

              train_loss /= len(self.train_data)
              train_acc = train_correct / len(self.train_data)
                  
          self.model.eval()
          with torch.no_grad():
                  val_outputs = self.model(self.private_test_X.to(self.device))
                  _, val_predicted = torch.max(val_outputs.data, 1)

                  val_predicted = val_predicted.cpu()
                  y_test = self.private_test_y.cpu()
                  val_acc = (val_predicted == self.private_test_y).sum().item() / len(self.private_test_X)


                  
                  if train_acc > val_acc + 0.20:
                    print("Train accuracy is 0.20 larger than validation accuracy. Stopping training.")
                    # Save the model
                    should_load = True
                    torch.save(self.model.state_dict(), "temp_saved.pth")
                    break

                  if val_acc == 0 or (val_acc > best_val_acc + min_delta):
                    best_val_acc = val_acc
                    patience_counter = 0
                  else:
                    patience_counter += 1
                    if patience_counter >= patience:
                      print("Early stopping triggered.")
                      break

                  if verbose > 0:
                    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

      if should_load:
        self.model.load_state_dict(torch.load("temp_saved.pth"))
      
  
    def communicateStep(self, public_dataset, batch_size):
        cd = CustomDataset(public_dataset)
        dl = torch.utils.data.DataLoader(cd, batch_size=batch_size, shuffle=False, num_workers=self.num_workers) if public_dataset.__len__() != 0 else None

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for j, data in enumerate(dl):

                input_data_tensor, target_data_tensor = data[0].to(self.device), data[1].to(self.device)

                # Forward pass
                outputs = self.model(input_data_tensor)

                #_, predicted = torch.max(outputs.data, 1)
                predictions.extend(outputs.cpu().numpy())
              
        return np.array(predictions)
      
    def digest(self, consensus, public_dataset, batch_size, num_epochs):

        consensus_tensor = torch.tensor(consensus)
        consensus_softmax = F.softmax(consensus_tensor, dim=1)
        consensus_labels = torch.argmax(consensus_softmax, dim=1)

        label_counts = torch.bincount(consensus_labels)
        print(f"labelle {label_counts}")

        dataset = ConsensusDataset(public_dataset, consensus_labels)
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        self.trainingMD(num_epochs = num_epochs, dataloader = dl)


    def revisit(self, num_epochs, batch_size):
        dl = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        self.trainingMD(num_epochs = num_epochs, dataloader = dl)


    def trainingMD(self, num_epochs=1, dataloader = None):
      criterion = nn.CrossEntropyLoss().to(self.device)

      optimizer = optim.Adam(self.model.parameters(), lr=0.001)

      losses = np.empty(num_epochs)

      for epoch in range(num_epochs):
        self.model.train()
        losses[epoch] = self.runEpochMD(dataloader, optimizer, criterion)
      self.losses = losses


    def runEpochMD(self, dl, optimizer, criterion):
      running_loss = 0.0
      i = 0
      for j, data in enumerate(dl):
          input_data_tensor, target_data_tensor = data[0].to(self.device), data[1].to(self.device)

          optimizer.zero_grad()
          outputs = self.model(input_data_tensor)
          loss = criterion(outputs, target_data_tensor)
          loss.backward()
          running_loss += loss.item()
          optimizer.step()

          i += 1

      if i == 0:
        print("Not running epoch", self.id)
        return 0
      return running_loss / i


    def evaluateFEDMD(self):
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        for data in self.testloader:
            input_tensor, labels_tensor = data[0].to(self.device), data[1].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                test_loss += F.cross_entropy(outputs, labels_tensor, reduction='sum').item()
                _, predicted = torch.max(outputs.data, 1)  # same as torch.argmax()
                total += labels_tensor.size(0)
                correct += (predicted == labels_tensor).sum().item()
        if total == 0:
            accuracy = 0
            test_loss = 0
        else:
            accuracy = 100 * correct / total
            test_loss /= total
        return accuracy, test_loss
   
 
    def test(self, batch_size, set_to_use='test'):
        """Tests self.model on self.test_data.

        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            dataloader = self.trainloader
        elif set_to_use == 'test' or set_to_use == 'val':
            dataloader = self.testloader

        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        for data in dataloader:
            input_tensor, labels_tensor = data[0].to(self.device), data[1].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                test_loss += F.cross_entropy(outputs, labels_tensor, reduction='sum').item()
                _, predicted = torch.max(outputs.data, 1)  # same as torch.argmax()
                total += labels_tensor.size(0)
                correct += (predicted == labels_tensor).sum().item()
        if total == 0:
            accuracy = 0
            test_loss = 0
        else:
            accuracy = 100 * correct / total
            test_loss /= total
        return {ACCURACY_KEY: accuracy, 'loss': test_loss}

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        return self.eval_data.__len__()

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        return self.train_data.__len__()

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        return self.num_train_samples + self.num_test_samples

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model

    def total_grad_norm(self):
        """Returns L2-norm of model total gradient"""
        total_norm = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except Exception:
                    # this param had no grad
                    pass
        total_norm = total_norm ** 0.5
        return total_norm

    def params_norm(self):
        """Returns L2-norm of client's model parameters"""
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def lr_scheduler_step(self, step):
        """Update learning rate according to given step"""
        self.lr *= step

    def update_lr(self, lr):
        self.lr = lr

    def _client_labels(self):
        """Returns client labels (only for analysis purposes)"""
        labels = set()
        if self.train_data.__len__() > 0:
            loader = self.trainloader
        else:
            loader = self.testloader
        for data in loader:
            l = data[1].tolist()
            labels.update(l)
        return list(labels)

    def number_of_samples_per_class(self):
        if self.train_data.__len__() > 0:
            loader = self.trainloader
        else:
            loader = self.testloader
        samples_per_class = {}
        for data in loader:
            labels = data[1].tolist()
            for l in labels:
                if l in samples_per_class:
                    samples_per_class[l] += 1
                else:
                    samples_per_class[l] = 1
        return samples_per_class

    def get_task_info(self):
        """Returns client's task (only for analysis purposes)"""
        return self._classes.copy()