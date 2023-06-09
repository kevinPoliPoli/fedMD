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

class Client:

    def __init__(self, seed, client_id, lr, weight_decay, batch_size, momentum, train_data, eval_data, model, public_dataset, public_test_dataset, device=None,
                 num_workers=0, run=None, mixup=False, mixup_alpha=1.0):
        self._model = model
        self.id = client_id
        self.train_data = train_data
        self.eval_data = eval_data
        self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers) if self.train_data.__len__() != 0 else None
        self.testloader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers) if self.eval_data.__len__() != 0 else None
        self._classes = self._client_labels()
        self.num_samples_per_class = self.number_of_samples_per_class()
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

        self.public_dataset = public_dataset  
        self.public_test_dataset = public_test_dataset
        self.public_test_loader = torch.utils.data.DataLoader(public_test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) if self.public_test_dataset.__len__() != 0 else None
    

    def transferLearningInit(self, num_epochs=25, batch_size=32):    
      dlTL = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers) if self.train_data.__len__() != 0 else None
      self.trainingMD(num_epochs=num_epochs, batch_size = batch_size, dl = dlTL)


    def communicate(self, batch_size = 256):
        dlCommunicate = torch.utils.data.DataLoader(self.public_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers) if self.public_dataset.__len__() != 0 else None
        self.communicateStep(dlCommunicate)
  
    def communicateStep(self, dl = None):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for j, data in enumerate(dl):
                
                input_data_tensor, target_data_tensor = data[0].to(self.device), data[1].to(self.device)

                # Forward pass
                outputs = self.model(input_data_tensor)
        
                #_, predicted = torch.max(outputs.data, 1)
                predictions.extend(outputs.cpu().numpy())
              
        return predictions
      
    def digest(self, consensus, batch_size = 256, num_epochs = 1):
      consensus_tensor = torch.tensor(consensus)
      dlDigest = torch.utils.data.DataLoader(self.public_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers) if self.public_dataset.__len__() != 0 else None
      self.trainingMD(consensus_tensor, Digest = True, num_epochs = num_epochs, batch_size = batch_size,  dl = dlDigest)


    def revisit(self, batch_size = 5, num_epochs = 4):
      dlRevisit = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers) if self.train_data.__len__() != 0 else None
      self.trainingMD(Revisit = True, num_epochs = num_epochs, batch_size = batch_size,  dl = dlRevisit)


    def trainingMD(self, consensus = None, Digest=False, Revisit=False, num_epochs=1, batch_size=64, dl = None):
      criterion = nn.CrossEntropyLoss().to(self.device)
      optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
      losses = np.empty(num_epochs)

      for epoch in range(num_epochs):
        self.model.train()
        
        if Digest:
          losses[epoch] = self.runEpochMD(dl, optimizer, criterion, Digest=True, consensus=consensus, batch_size = batch_size)
    
        elif Revisit:
          losses[epoch] = self.runEpochMD(dl, optimizer, criterion, Revisit=True, batch_size = batch_size)

        else:
          losses[epoch] = self.runEpochMD(dl, optimizer, criterion, batch_size = batch_size)
      
      self.losses = losses

    def runEpochMD(self, dl, optimizer, criterion, Digest = False, Revisit = False, consensus = None, batch_size = 64):
      running_loss = 0.0
      i = 0
      for j, data in enumerate(dl):
          input_data_tensor, target_data_tensor = data[0].to(self.device), data[1].to(self.device)

          optimizer.zero_grad()
          outputs = self.model(input_data_tensor)

          if Digest:
            start_index = j * batch_size
            end_index = (j + 1) * batch_size
      
            consensus_batch = consensus[start_index:end_index].to(self.device)
            _, consensus_batch_labels = torch.max(consensus_batch, 1)

            loss = criterion(outputs, consensus_batch_labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

          else:
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
        for data in self.public_test_loader:
            input_tensor, labels_tensor = data[0].to(self.device), data[1].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                test_loss += F.cross_entropy(outputs, labels_tensor, reduction='sum').item()
                _, predicted = torch.max(outputs.data, 1)  
                total += labels_tensor.size(0)
                correct += (predicted == labels_tensor).sum().item()
        if total == 0:
            accuracy = 0
            test_loss = 0
        else:
            accuracy = 100 * correct / total
            test_loss /= total
        return accuracy, test_loss

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