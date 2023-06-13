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
    
    
      
    #compute upperbound
    """
    from architecturesMD.upperbounds import start_upperbound
    print("UpperBound")
    start_upperbound(c_models, model_names, device)
    """
    #### Create server ####
    
    server_p = MODEL_PARAMS[server_model_path]
    server_model = ServerModel(*server_p, device)
    server_model = server_model
    server = Server(server_model)

    #### Create client datasets ####
    from utils import create_datasets_md as cd
    
    private_classes = [0,2,20,63,71,82]

    CIFAR10_images, CIFAR10_labels = cd.load_CIFAR10(train=True)
    CIFAR10_images_test, CIFAR10_labels_test = cd.load_CIFAR10(train=False)
    public_dataset = {"X": CIFAR10_images, "y": CIFAR10_labels}
    
    CIFAR100_train_X, CIFAR100_train_y = cd.load_CIFAR100(train=True)
    CIFAR100_test_X, CIFAR100_test_y = cd.load_CIFAR100(train=False)
    CIFAR100_X, CIFAR100_Y = cd.generate_partial_data(CIFAR100_train_X, CIFAR100_train_y, private_classes, verbose=True)
    CIFAR100_X_test, CIFAR100_Y_test = cd.generate_partial_data(CIFAR100_test_X, CIFAR100_test_y, private_classes, verbose=True)
    
    CIFAR100_Y = np.array(CIFAR100_Y)
    CIFAR100_Y_test = np.array(CIFAR100_Y_test)

    for index, cls_ in enumerate(private_classes):        
        CIFAR100_Y[CIFAR100_Y == cls_] = index + 10
        CIFAR100_Y_test[CIFAR100_Y_test == cls_] = index + 10
    del index, cls_
    
    
    private_data, total_private_data = cd.generate_bal_private_data(CIFAR100_X, CIFAR100_Y,      
                               N_parties = 10,           
                               classes_in_use = np.arange(6) + 10, 
                               N_samples_per_class = 3, 
                               data_overlap = False)

    X_tmp, y_tmp = cd.generate_partial_data(X = CIFAR100_X_test, y= CIFAR100_Y_test,
                                         class_in_use = np.arange(6) + 10, 
                                         verbose = True)
    
    private_test_data = {"X": X_tmp, "y": y_tmp}
    del X_tmp, y_tmp
    
    from architecturesMD.cnn import _returnModel, train_models

    #our resnet
    resnet_model_path = 'cifar100.resnet'
    mod = importlib.import_module(resnet_model_path)
    ClientModel = getattr(mod, 'ClientModel')    
    model_params = MODEL_PARAMS[resnet_model_path]
    custom_resnet20 = ClientModel(*model_params, device)

    #create client models
    models_dictionary = [{"model_name": "2_layer_CNN", "params": {"n1": 128, "n2": 256, "dropout_rate": 0.2}},
               {"model_name": "2_layer_CNN", "params": {"n1": 128, "n2": 384, "dropout_rate": 0.2}},
               {"model_name": "2_layer_CNN", "params": {"n1": 128, 'n2': 512, "dropout_rate": 0.2}},
               {"model_name": "2_layer_CNN", "params": {"n1": 256, "n2": 256, "dropout_rate": 0.3}},
               {"model_name": "2_layer_CNN", "params": {"n1": 256, "n2": 512, "dropout_rate": 0.4}},
               {"model_name": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 256, "dropout_rate": 0.2}},
               {"model_name": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.2}},
               {"model_name": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 128, "dropout_rate": 0.3}},
               {"model_name": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
              ]
    
    c_models = []
    c_models.append(custom_resnet20)

    model_saved_names = ["Custom_Resnet20", "CNN_128_256", "CNN_128_384", "CNN_128_512", "CNN_256_256", "CNN_256_512", 
                    "CNN_64_128_256", "CNN_64_128_192", "CNN_128_128_128", "CNN_128_128_192"]
                    
    for i, item in enumerate(models_dictionary):
        model_name = item["model_name"]
        model_params = item["params"]
        tmp = _returnModel(model_name, n_classes=16, 
                                        input_shape=(32,32,3),
                                        **model_params)
        print("model {0} : {1}".format(i, model_saved_names[i+1]))
        c_models.append(tmp)
    
    del model_name, model_params, tmp
    
    pre_train_params = {"min_delta": 0.005, "patience": 3,
                     "batch_size": 128, "epochs": 20, "is_shuffle": True, 
                     "verbose": 1}
    
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
          "CNN_128_256": c_models[1],
          "CNN_128_384": c_models[2],
          "CNN_128_512": c_models[3],
          "CNN_256_256": c_models[4],
          "CNN_256_512": c_models[5],
          "CNN_64_128_256": c_models[6], 
          "CNN_64_128_192": c_models[7],
          "CNN_128_128_128": c_models[8],
          "CNN_128_128_192":c_models[9]
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
            
     
    del CIFAR10_images, CIFAR10_labels, CIFAR10_images_test, CIFAR10_labels_test, CIFAR100_X, CIFAR100_Y, CIFAR100_X_test, CIFAR100_Y_test

    #### Start Experiment ####
    start_round = 0
    print("Start round:", start_round)

    if args.pretrained:
      train_clients = create_clients(np.arange(10), private_data, total_private_data, c_models, args, Client, run=None, device=device)
    else:
      train_clients = create_clients(np.arange(10), private_data, total_private_data, client_models, args, Client, run=None, device=device)


    train_client_ids, train_client_num_samples = server.get_clients_info(train_clients)
    print('Clients in Total: %d' % len(train_clients))    
    server.set_num_clients(len(train_clients))

    
    for c in train_clients:
      c.transferLearningInit()


    # Start training
    for i in range(start_round, num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        public_dataset_round = cd.generate_alignment_data(public_dataset['X'], public_dataset['y'], N_alignment = 5000)

        # Select clients to train during this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)
        print("Selected clients:", c_ids)

    
        _ = server.train_model(num_epochs_digest = 1, num_epochs_revisit = 4, batch_size_digest=256, batch_size_revisit = 5, public_dataset = public_dataset_round, device = device)
  
def online(clients):
    return clients

def create_clients(train_users, private_data, total_private_data, models, args, Client, run=None, device=None):


    clients = []
    client_params = define_client_params(args.client_algorithm, args)

    client_params['run'] = run
    client_params['device'] = device
    c_testdata = total_private_data
    client_params['eval_data'] = c_testdata
  

    for u in train_users:
        client_params['model'] = models[u]
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
