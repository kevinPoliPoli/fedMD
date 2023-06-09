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

    c_models = [resnet32, resnet44, resnet56]
    
  
    server_p = MODEL_PARAMS[server_model_path]
    server_model = ServerModel(*server_p, device)
    server_model = server_model.to(device)

        
    #### Create server ####
    server_params = define_server_params(args, server_model, args.algorithm, opt_ckpt=None)
    server = Server(**server_params)

    start_round = 0
    print("Start round:", start_round)

    train_clients, test_clients = setup_clients(args, c_models, Client, ClientDataset, PublicDataset, None, device)
    train_client_ids, train_client_num_samples = server.get_clients_info(train_clients)
    test_client_ids, test_client_num_samples = server.get_clients_info(test_clients)
    if set(train_client_ids) == set(test_client_ids):
        print('Clients in Total: %d' % len(train_clients))
    else:
        print(f'Clients in Total: {len(train_clients)} training clients and {len(test_clients)} test clients')

    server.set_num_clients(len(train_clients))

    for c in train_clients:
      print("initializing client: " + c.id)
      c.transferLearningInit()

    # Start training
    for i in range(start_round, num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))


        # Select clients to train during this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)
        print("Selected clients:", c_ids)

    
        _ = server.train_model(num_epochs_digest=args.num_epochs_digest, num_epochs_revisit=args.num_epochs_revisit, 
                               batch_size_digest=args.batch_size_digest, batch_size_revisit=args.batch_size_revisit)


        print("Evaluating clients")
        server.evaluateClients()
       


def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, train_data, test_data, models, args, ClientDataset, Client, public_data, public_test_data, PublicDataset, users_p, users_pt, run=None, device=None):

    import random
    import copy

    clients = []
    client_params = define_client_params(args.client_algorithm, args)

    client_params['run'] = run
    client_params['device'] = device
    client_params['public_dataset'] = PublicDataset(public_data, users_p, public_dataset = True, train=True, loading=args.where_loading, cutout=Cutout if args.cutout else None)
    client_params['public_test_dataset'] = PublicDataset(public_test_data, users_pt, public_dataset = True, train=False, loading=args.where_loading, cutout=Cutout if args.cutout else None)
    
    for u in users:
        model = random.choice(models)
        client_params['model'] = copy.deepcopy(model)

        c_traindata = ClientDataset(train_data[u], train=True, loading=args.where_loading, cutout=Cutout if args.cutout else None)
        c_testdata = ClientDataset(test_data[u], train=False, loading=args.where_loading, cutout=None)
        client_params['client_id'] = u
        client_params['train_data'] = c_traindata
        client_params['eval_data'] = c_testdata
        clients.append(Client(**client_params))
    return clients


def setup_clients(args, models, Client, ClientDataset, PublicDataset, run=None, device=None,):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    
    train_data_dir = os.path.join('..', 'data', 'cifar100', 'data', 'train')
    test_data_dir = os.path.join('..', 'data', 'cifar100', 'data', 'test')
    
    public_data_dir = os.path.join('..', 'data', 'cifar10', 'data', 'train')
    public_test_data_dir = os.path.join('..', 'data', 'cifar10', 'data', 'test')
    
    

    train_users, _, test_users, _, train_data, test_data = read_data(train_data_dir, test_data_dir, args.alpha)
    users_p, _, users_pt, _, public_data, public_test_data = read_data(public_data_dir, public_test_data_dir, 100)
    
    
    train_clients = create_clients(train_users, train_data, test_data, models, args, ClientDataset, Client, public_data, public_test_data, PublicDataset, users_p, users_pt, run, device)
    test_clients = create_clients(test_users, train_data, test_data, models, args, ClientDataset, Client, public_data, public_test_data, PublicDataset, users_p, users_pt, run, device)

    return train_clients, test_clients

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
