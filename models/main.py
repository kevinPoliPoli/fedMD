"""Script to run the baselines."""
import importlib
import inspect
import json
import numpy as np
import os
import pandas as pd
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import random
import torch
import torch.nn as nn
from datetime import datetime

import metrics.writer as metrics_writer
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, ACCURACY_KEY, CLIENT_PARAMS_KEY, CLIENT_GRAD_KEY, \
    CLIENT_TASK_KEY
from utils.args import parse_args, check_args
from utils.cutout import Cutout
from utils.main_utils import *
from utils.model_utils import read_data

def main():
    args = parse_args()
    check_args(args)

    # Set the random seed if provided (affects client sampling and batching)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # CIFAR: obtain info on parameter alpha (Dirichlet's distribution)
    alpha = args.alpha
    if alpha is not None:
        alpha = 'alpha_{:.2f}'.format(alpha)
        print("Alpha:", alpha)

    # Setup GPU
    device = torch.device(args.device if torch.cuda.is_available else 'cpu')
    print("Using device:", torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu')

    # Obtain the path to client's model (e.g. cifar10/cnn.py), client class and servers class
    public_dataset_path = 'cifar10/dataloader.py'
    private_dataset_path = 'cifar100/dataloader.py'

    server_path = 'servers/%s.py' % (args.algorithm + '_server')
    client_path = 'clients/%s.py' % (args.client_algorithm + '_client' if args.client_algorithm is not None else 'client')
    check_init_paths([public_dataset_path, private_dataset_path, server_path, client_path])

    model_path = '%s.%s' % (args.dataset, args.model)

    public_dataset_path = 'cifar10.dataloader'
    private_dataset_path = 'cifar100.dataloader'
    
    server_model_path = 'cifar10.resnet'
  
    server_path = 'servers.%s' % (args.algorithm + '_server')
    client_path = 'clients.%s' % (args.client_algorithm + '_client' if args.client_algorithm is not None else 'client')

    server_mod = importlib.import_module(server_model_path)
    ServerModel = getattr(server_mod, 'ClientModel')
    pub_dataset = importlib.import_module(public_dataset_path)
    PublicDataset = getattr(pub_dataset, 'ClientDataset')
    
  
    # Load client and server
    print("Running experiment with server", server_path, "and client", client_path)
    Client, Server = get_client_and_server(server_path, client_path)
    print("Verify client and server:", Client, Server)

    # Experiment parameters (e.g. num rounds, clients per round, lr, etc)
    tup = MAIN_PARAMS['cifar100'][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]
    

    #create client models
    dataset = importlib.import_module(private_dataset_path)
    ClientDataset = getattr(dataset, 'ClientDataset')

    from architecturesMD.resnets import _resnet
    resnet32 = _resnet([5]*3, pretrained=True, type=32).to(device)
    resnet44 = _resnet([7]*3, pretrained=True, type=44).to(device)
    resnet56 = _resnet([9]*3, pretrained=True, type=56).to(device)

    #add other nets
    c_models = [resnet32, resnet44, resnet56]


    #compute upperbound
    print("Diocane")
    test_loader = load_full_test([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    for model in c_models:
      calculate_upperbound(model, test_loader, device)
        

    
    
    server_p = MODEL_PARAMS[server_model_path]
    server_model = ServerModel(*server_p, device)
    server_model = server_model.to(device)

        
    #### Create server ####
    server_params = define_server_params(args, server_model, args.algorithm, opt_ckpt=None)
    server = Server(**server_params)

    start_round = 0
    print("Start round:", start_round)
  
    train_clients = setup_clients(args, c_models, Client, ClientDataset, None, device)
    train_client_ids, train_client_num_samples = server.get_clients_info(train_clients)
    print('Clients in Total: %d' % len(train_clients))
    
    server.set_num_clients(len(train_clients))

    print("prima")
    server.evaluateClients(train_clients)
    for c in train_clients:
      print("initializing client: " + c.id)
      c.transferLearningInit()

    print("dopo")

    server.evaluateClients(train_clients)

    public_data_dir = os.path.join('..', 'data', 'cifar10', 'data', 'train')
    public_test_data_dir = os.path.join('..', 'data', 'cifar10', 'data', 'test')

    # Start training
    for i in range(start_round, num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        users_p, _, _, _, public_data, _ = read_data(public_data_dir, public_test_data_dir, 100)
        public_dataset_round = PublicDataset(public_data, users_p, public_dataset = True, train=True, loading=args.where_loading, cutout=Cutout if args.cutout else None)


        # Select clients to train during this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)
        print("Selected clients:", c_ids)

    
        _ = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, public_dataset = public_dataset_round)

        server.evaluateClients()
       

        ##### Update server model (FedAvg) #####
        print("--- Updating central model ---")
        server.update_model()



def load_full_test(target_labels):
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

    # Filter the dataset based on target_labels
    filtered_indices = [idx for idx, label in enumerate(test_dataset.targets) if label in target_labels]
    filtered_dataset = torch.utils.data.Subset(test_dataset, filtered_indices)

    test_loader = torch.utils.data.DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

def calculate_upperbound(model, test_loader, device):
  import torch.optim as optim

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0004)

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

def create_clients(train_users, test_users, train_data, test_data, models, args, ClientDataset, Client, run=None, device=None):

    import random
    import copy
    clients = []
    client_params = define_client_params(args.client_algorithm, args)


    client_params['run'] = run
    client_params['device'] = device
    c_testdata = ClientDataset(test_data, users = test_users, aggregated_test = True, train=False, loading=args.where_loading, cutout=None)
    client_params['eval_data'] = c_testdata
    
    participants = train_users[0:10]
    for u in participants:
        model = random.choice(models)
        client_params['model'] = copy.deepcopy(model)
        c_traindata = ClientDataset(train_data[u], train=True, loading=args.where_loading, cutout=Cutout if args.cutout else None)
        client_params['client_id'] = u
        client_params['train_data'] = c_traindata
        
        clients.append(Client(**client_params))
    return clients


def setup_clients(args, models, Client, ClientDataset, run=None, device=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    
    train_data_dir = os.path.join('..', 'data', 'cifar100', 'data', 'train')
    test_data_dir = os.path.join('..', 'data', 'cifar100', 'data', 'test')
    
    train_users, train_groups, test_users, test_groups, train_data, test_data = read_data(train_data_dir, test_data_dir, args.alpha)
    train_clients = create_clients(train_users, test_users, train_data, test_data, models, args, ClientDataset, Client, run, device)

    return train_clients

def get_client_and_server(server_path, client_path):
    mod = importlib.import_module(server_path)
    cls = inspect.getmembers(mod, inspect.isclass)
    server_name = server_path.split('.')[1].split('_server')[0]
    server_name = list(map(lambda x: x[0], filter(lambda x: 'Server' in x[0] and server_name in x[0].lower(), cls)))[0]
    Server = getattr(mod, server_name)
    mod = importlib.import_module(client_path)
    cls = inspect.getmembers(mod, inspect.isclass)
    client_name = max(list(map(lambda x: x[0], filter(lambda x: 'Client' in x[0], cls))), key=len)
    Client = getattr(mod, client_name)
    return Client, Server


if __name__ == '__main__':
    main()
