"""Script to run the baselines."""
import importlib
import inspect
import json
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
from datetime import datetime

from servers.fedavg_server import Server

import metrics.writer as metrics_writer
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, ACCURACY_KEY, CLIENT_PARAMS_KEY, CLIENT_GRAD_KEY, \
    CLIENT_TASK_KEY
from utils.args import parse_args, check_args
from utils.cutout import Cutout
from utils.main_utils import *
from utils.model_utils import read_data
from utils.custom_dataloader import CustomDataset
from utils import create_datasets_md as cd
from architecturesMD.cnn import _returnModel, train_models
from architecturesMD.resnets import _resnet

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

    server_path = 'servers/%s.py' % (args.algorithm + '_server')
    client_path = 'clients/%s.py' % (args.client_algorithm + '_client' if args.client_algorithm is not None else 'client')
    model_path = '%s.%s' % (args.dataset, args.model)

    
    client_path = 'clients.%s' % (args.client_algorithm + '_client' if args.client_algorithm is not None else 'client')

  
    # Load client and server
    Client = get_client(client_path)

    # Experiment parameters (e.g. num rounds, clients per round, lr, etc)
    tup = MAIN_PARAMS['cifar100'][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]


    #### Create server ####
    server = Server()

    #### CIFAR10 ###
    CIFAR10_images, CIFAR10_labels = cd.load_CIFAR10(train=True)
    CIFAR10_images_test, CIFAR10_labels_test = cd.load_CIFAR10(train=False)
    public_dataset = {"X": CIFAR10_images, "y": CIFAR10_labels}
    
    
    ###CLIENT MODELS###
    c_models = []
    #our resnet
    resnet_model_path = 'cifar100.resnet'
    mod = importlib.import_module(resnet_model_path)
    ClientModel = getattr(mod, 'ClientModel')    
    model_params = MODEL_PARAMS[resnet_model_path]
    custom_resnet20 = ClientModel(*model_params, device)

  
    #other resnets
    resnet18 = _resnet("resnet18", [3]*3)
    resnet32 = _resnet("resnet32", [5]*3)
    resnet36 = _resnet("resnet36", [6]*3)
    resnet44 = _resnet("resnet44", [7]*3)
    resnet56 = _resnet("resnet56", [9]*3)
    resnet60 = _resnet("resnet60", [10]*3)
    resnet72 = _resnet("resnet72", [12]*3)
    resnet84 = _resnet("resnet84", [14]*3)
    resnet96 = _resnet("resnet96", [16]*3)

    c_models.append(custom_resnet20)
    c_models.append(resnet18)
    c_models.append(resnet32)
    c_models.append(resnet36)
    c_models.append(resnet44)
    c_models.append(resnet56)
    c_models.append(resnet60)
    c_models.append(resnet72)
    c_models.append(resnet84)
    c_models.append(resnet96)

    ### PRETRAIN ###
    pre_train_params = {"min_delta": 0.005, "patience": 3,
                     "batch_size": 128, "epochs": 20, "is_shuffle": True, 
                     "verbose": 1}
    
    model_saved_names = ["Custom_Resnet20", "Resnet18", "Resnet32", "Resnet36", "Resnet44", "Resnet56", "Resnet60", "Resnet72", "Resnet84", "Resnet96"]
    client_models = []
    if args.pretrained:
        pre_train_result = train_models(c_models, 
                                        CIFAR10_images, CIFAR10_labels, 
                                        CIFAR10_images_test, CIFAR10_labels_test, device,
                                        save_dir="./model_weights", save_names=model_saved_names,
                                        early_stopping = True,
                                        **pre_train_params)
    else:
        model_dict = {
          "Custom_Resnet20": c_models[0],
          "Resnet18": c_models[1],
          "Resnet32": c_models[2],
          "Resnet36": c_models[3],
          "Resnet44": c_models[4],
          "Resnet56": c_models[5],
          "Resnet60": c_models[6],
          "Resnet72": c_models[7],
          "Resnet84": c_models[8],
          "Resnet96": c_models[9], 
        }

        dpath = os.path.abspath("./model_weights")
        model_names = os.listdir(dpath)

        for model in model_names:
            tmp = None
            model_path = os.path.join(dpath, model)
            state_dict = torch.load(model_path)

            name = os.path.splitext(model)[0]
            print(f"Loading model: {name}")
            model_dict[name].load_state_dict(state_dict)
            client_models.append(model_dict[name].to(device))
            
    del CIFAR10_images, CIFAR10_labels

    #create client private datasets
    private_classes = [0,1,7,9,12,18]

    CIFAR100_train_X, CIFAR100_train_y = cd.load_CIFAR100(train=True)
    CIFAR100_test_X, CIFAR100_test_y = cd.load_CIFAR100(train=False)
    
    _, y_train_super = cd.load_CIFAR100(train=True, label_type="coarse")
    _, y_test_super  = cd.load_CIFAR100(train=False, label_type="coarse")
    
    
    relations = [set() for i in range(np.max(y_train_super)+1)]
    for i, y_fine in enumerate(CIFAR100_train_y):
        relations[y_train_super[i]].add(y_fine)
    for i in range(len(relations)):
        relations[i]=list(relations[i])
    
    del i, y_fine
    
    fine_classes_in_use = [[relations[j][i%5] for j in private_classes] 
                           for i in range(10)]
    print(fine_classes_in_use)
    
    #Generate test set
    X_tmp, y_tmp = cd.generate_partial_data(CIFAR100_test_X, y_test_super,
                                         class_in_use = private_classes,
                                         verbose = True)
    
    
    for index in range(len(private_classes)-1, -1, -1):
        cls_ = private_classes[index]
        y_tmp[y_tmp == cls_] = index + 10
    private_test_data = {"X": X_tmp, "y": y_tmp}
    del index, cls_, X_tmp, y_tmp
    
    
    private_data, total_private_data = cd.generate_imbal_CIFAR_private_data(
                                        CIFAR100_train_X, CIFAR100_train_y, y_train_super,   
                                        N_parties = 10,   
                                        classes_per_party = fine_classes_in_use,
                                        samples_per_class = 20)
    
    
    for index in range(len(private_classes)-1, -1, -1):
        cls_ = private_classes[index]
        total_private_data["y"][total_private_data["y"] == cls_] = index + 10
        for i in range(10):
            private_data[i]["y"][private_data[i]["y"] == cls_] = index + 10
    
    del index, cls_
    
    
        
    ###CLIENT MODELS###
    c_models = []
    #our resnet
    resnet_model_path = 'cifar100.resnet'
    mod = importlib.import_module(resnet_model_path)
    ClientModel = getattr(mod, 'ClientModel')    
    model_params = MODEL_PARAMS[resnet_model_path]
    custom_resnet20 = ClientModel(*model_params, device)

  
    #other resnets
    resnet18 = _resnet("resnet18", [3]*3)
    resnet32 = _resnet("resnet32", [5]*3)
    resnet36 = _resnet("resnet36", [6]*3)
    resnet44 = _resnet("resnet44", [7]*3)
    resnet56 = _resnet("resnet56", [9]*3)
    resnet60 = _resnet("resnet60", [10]*3)
    resnet72 = _resnet("resnet72", [12]*3)
    resnet84 = _resnet("resnet84", [14]*3)
    resnet96 = _resnet("resnet96", [16]*3)

    c_models.append(custom_resnet20)
    c_models.append(resnet18)
    c_models.append(resnet32)
    c_models.append(resnet36)
    c_models.append(resnet44)
    c_models.append(resnet56)
    c_models.append(resnet60)
    c_models.append(resnet72)
    c_models.append(resnet84)
    c_models.append(resnet96)

    ### PRETRAIN ###
    pre_train_params = {"min_delta": 0.005, "patience": 3,
                     "batch_size": 128, "epochs": 20, "is_shuffle": True, 
                     "verbose": 1}
    
    model_saved_names = ["Custom_Resnet20", "Resnet18", "Resnet32", "Resnet36", "Resnet44", "Resnet56", "Resnet60", "Resnet72", "Resnet84", "Resnet96"]
    client_models = []
    if args.pretrained:
        pre_train_result = train_models(c_models, 
                                        CIFAR10_images, CIFAR10_labels, 
                                        CIFAR10_images_test, CIFAR10_labels_test, device,
                                        save_dir="./model_weights", save_names=model_saved_names,
                                        early_stopping = True,
                                        **pre_train_params)
    else:
        model_dict = {
          "Custom_Resnet20": c_models[0],
          "Resnet18": c_models[1],
          "Resnet32": c_models[2],
          "Resnet36": c_models[3],
          "Resnet44": c_models[4],
          "Resnet56": c_models[5],
          "Resnet60": c_models[6],
          "Resnet72": c_models[7],
          "Resnet84": c_models[8],
          "Resnet96": c_models[9], 
        }

        dpath = os.path.abspath("./model_weights")
        model_names = os.listdir(dpath)

        for model in model_names:
            tmp = None
            model_path = os.path.join(dpath, model)
            state_dict = torch.load(model_path)

            name = os.path.splitext(model)[0]
            print(f"Loading model: {name}")
            model_dict[name].load_state_dict(state_dict)
            client_models.append(model_dict[name].to(device))
            
    start_round = 0
    print("Start round:", start_round)

    if args.pretrained:
      train_clients = create_clients(np.arange(10), private_data, total_private_data, private_test_data, c_models, args, Client, run=None, device=device)
    else:
      train_clients = create_clients(np.arange(10), private_data, total_private_data, private_test_data, client_models, args, Client, run=None, device=device)


    train_client_ids, train_client_num_samples = server.get_clients_info(train_clients)
    print('Clients in Total: %d' % len(train_clients))    
    server.set_num_clients(len(train_clients))


    model_parameters = {
        "resnet18": {"lr" :0.001, "weight_decay" : 0.004},
        "resnet20": {"lr" :0.001, "weight_decay" : 0.004},
        "resnet32": {"lr" :0.001, "weight_decay" : 0.004},
        "resnet36": {"lr" :0.001, "weight_decay" : 0.004},
        "resnet44": {"lr" :0.001, "weight_decay" : 0.004},
        "resnet56": {"lr" :0.001, "weight_decay" : 0.004},
        "resnet60": {"lr" :0.001, "weight_decay" : 0.004},
        "resnet72": {"lr" :0.001, "weight_decay" : 0.004},
        "resnet84": {"lr" :0.001, "weight_decay" : 0.004},
        "resnet96": {"lr" :0.001, "weight_decay" : 0.004},
      }

    
    for c in train_clients:
      c.transferLearningInit(model_parameters)

    import copy
    copied_models = []
    for c in train_clients:
      print(f"Copying {c.id} model {c.model.name} for upperbound")
      copied_models.append(copy.deepcopy(c.model))

    from architecturesMD.upperbounds import start_upperbound, plot_accuracy_epochs
    upperbounds = start_upperbound(total_private_data, private_test_data, copied_models, model_saved_names, device)
 
    accuracies = [[], [], [], [], [], [], [], [], [], []]

    # Start training
    for i in range(start_round, num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        public_dataset_round = cd.generate_alignment_data(public_dataset['X'], public_dataset['y'], N_alignment = 5000)

        # Select clients to train during this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)
        print("Selected clients:", c_ids)
    
        server.train_model(accuracies, num_epochs_digest = 1, num_epochs_revisit = 10, batch_size_digest=128, batch_size_revisit = 19, public_dataset = public_dataset_round, device = device)
  

    print(f'{num_rounds}')
    print(f'accuracies:{accuracies}, len:{len(accuracies[0])}')
    print(f'upperbounds: {upperbounds}, len:{len(upperbounds)}')
    plot_accuracy_epochs(num_rounds, accuracies, upperbounds)
  
def online(clients):
    return clients

def create_clients(train_users, private_data, total_private_data, private_test_data, models, args, Client, run=None, device=None):
    clients = []
    client_params = define_client_params(args.client_algorithm, args)

    client_params['run'] = run
    client_params['device'] = device
    client_params['eval_data'] = total_private_data
    client_params['private_test'] = private_test_data
  

    for u in train_users:
        client_params['model'] = models[u]
        c_traindata = private_data[u]
        client_params['client_id'] = u
        client_params['train_data'] = c_traindata
        
        clients.append(Client(**client_params))
    return clients


def get_client(client_path):
    mod = importlib.import_module(client_path)
    cls = inspect.getmembers(mod, inspect.isclass)
    client_name = max(list(map(lambda x: x[0], filter(lambda x: 'Client' in x[0], cls))), key=len)
    Client = getattr(mod, client_name)
    return Client

if __name__ == '__main__':
    main()