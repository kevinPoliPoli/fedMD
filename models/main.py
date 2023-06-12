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
from utils.custom_dataloader import CustomDataset

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
    
    
    #### Create client models ####

    from architecturesMD.resnets import _resnet

    resnet_model_path = 'cifar100.resnet'
    mod = importlib.import_module(resnet_model_path)
    ClientModel = getattr(mod, 'ClientModel')    
    model_params = MODEL_PARAMS[resnet_model_path]
    client_model = ClientModel(*model_params, device)
    custom_resnet20 = client_model.to(device)
    model_weights_path = './architecturesMD/cifar10_custom_resnet20.pth'
    
    print("Loading state dict of resnet20")
    state_dict = torch.load(model_weights_path)
    state_dict_without_fc = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
    custom_resnet20.load_state_dict(state_dict_without_fc, strict=False)
    
    custom_resnet20 = custom_resnet20.to(device)
    resnet32 = _resnet([5]*3, pretrained=True, type=32).to(device)
    resnet44 = _resnet([7]*3, pretrained=True, type=44).to(device)
    resnet56 = _resnet([9]*3, pretrained=True, type=56).to(device)

    #add other nets
    c_models = [custom_resnet20, resnet32, resnet44, resnet56]
    model_names = ["resnet20", "resnet32", "resnet44", "resnet56"]

    #compute upperbound
    """
    from architecturesMD.upperbounds import start_upperbound
    print("UpperBound")
    start_upperbound(c_models, model_names, device)
    """
  
  
    #### Create server ####
    server_p = MODEL_PARAMS[server_model_path]
    server_model = ServerModel(*server_p, device)
    server_model = server_model.to(device)
    server = Server(server_model)


    #### Create client datasets ####
    from utils import create_datasets_md as cd
    
    CIFAR100_images, CIFAR100_labels = cd.load_CIFAR100()
    private_data, total_private_data = cd.generate_bal_private_data(CIFAR100_images, CIFAR100_labels,      
                               N_parties = 10,           
                               classes_in_use = np.arange(6) + 10, 
                               N_samples_per_class = 3, 
                               data_overlap = False)

    
    CIFAR10_images, CIFAR10_labels = cd.load_CIFAR10()
    
    #### Start Experiment ####
    start_round = 0
    print("Start round:", start_round)
  
    train_clients = create_clients(np.arange(10), private_data, total_private_data, c_models, args, Client, run=None, device=device)
    train_client_ids, train_client_num_samples = server.get_clients_info(train_clients)
    print('Clients in Total: %d' % len(train_clients))    
    server.set_num_clients(len(train_clients))

    print("prima")
    server.evaluateClients(train_clients)

    for c in train_clients:
      c.transferLearningInit()

    print("dopo")
    server.evaluateClients(train_clients)

    # Start training
    for i in range(start_round, num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        public_dataset_round = cd.generate_alignment_data(CIFAR10_images, CIFAR10_labels, N_alignments = 5000)
        public_dataset_round = CustomDataset(public_dataset_round)
        
        # Select clients to train during this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)
        print("Selected clients:", c_ids)

    
        _ = server.train_model(num_epochs_digest = 1, num_epochs_revisit = 1, batch_size=64, public_dataset = public_dataset_round)

        server.evaluateClients()

def online(clients):
    """We assume all users are always online."""
    return clients

def create_clients(train_users, private_data, total_private_data, models, args, Client, run=None, device=None):

    import random
    import copy
    clients = []
    client_params = define_client_params(args.client_algorithm, args)


    client_params['run'] = run
    client_params['device'] = device
    c_testdata = total_private_data
    client_params['eval_data'] = c_testdata
  
    for u in train_users:
        model = random.choice(models)
        client_params['model'] = copy.deepcopy(model)
        c_traindata = private_data[u]
        client_params['client_id'] = u
        client_params['train_data'] = c_traindata
        
        clients.append(Client(**client_params))
    return clients


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
