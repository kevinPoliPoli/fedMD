import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
import torchvision.transforms as transforms


def generate_bal_private_data(X, y, N_parties=10, classes_in_use=range(11), 
                              N_samples_per_class=20, data_overlap=False):
    priv_data = [None] * N_parties
    combined_idx = np.array([], dtype=np.int16)
    y = np.array(y)  
    
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        idx = np.random.choice(idx, N_samples_per_class * N_parties, 
                               replace=data_overlap)
        combined_idx = np.r_[combined_idx, idx]
        
        for i in range(N_parties):           
            idx_tmp = idx[i * N_samples_per_class: (i + 1) * N_samples_per_class]
            if priv_data[i] is None:
                tmp = {}
                tmp["X"] = X[idx_tmp]
                tmp["y"] = y[idx_tmp]
                tmp["idx"] = idx_tmp
                priv_data[i] = tmp
            else:
                priv_data[i]['idx'] = np.r_[priv_data[i]["idx"], idx_tmp]
                priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]
                
    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    
    return priv_data, total_priv_data



def generate_alignment_data(X, y, N_alignment = 3000):
    
    split = StratifiedShuffleSplit(n_splits=1, train_size= N_alignment)
    y = np.array(y) 
    if N_alignment == "all":
        alignment_data = {}
        alignment_data["idx"] = np.arange(y.shape[0])
        alignment_data["X"] = X
        alignment_data["y"] = y
        return alignment_data
    for train_index, _ in split.split(X, y):
        X_alignment = X[train_index]
        y_alignment = y[train_index]
    alignment_data = {}
    alignment_data["idx"] = train_index
    alignment_data["X"] = X_alignment
    alignment_data["y"] = y_alignment
    
    return alignment_data
    
def load_CIFAR100(train=True, label_type="fine"):
    
    if train:
        transform = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                    ])

        ds = torchvision.datasets.CIFAR100(root='./private_data', train=True, download=True, transform=transform)

        if label_type == "coarse":
          coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
          ds.targets = coarse_labels[ds.targets]

    else:
        test_transform = transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                      ])
        
        ds = torchvision.datasets.CIFAR100(root='./private_data', train=False, download=True, transform=test_transform)

        if label_type == "coarse":
          coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
          ds.targets = coarse_labels[ds.targets]


    images = []
    labels = []

    for image, label in ds:
            images.append(np.array(image))  # Convert PIL image to NumPy array
            labels.append(label)

    images = np.array(images)
    return images, labels


def load_CIFAR10(train=True):

  if train:
    transform = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  ])

   
    ds = torchvision.datasets.CIFAR10(root='./public_data', train=True, download=True, transform=transform)
  
  else: 
    transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])

    ds = torchvision.datasets.CIFAR10(root='./public_data', train=False, download=True, transform=transform)


  images = []
  labels = []

  for image, label in ds:
      images.append(np.array(image))  # Convert PIL image to NumPy array
      labels.append(label)

  images = np.array(images)
  return images, labels
    
    
def generate_partial_data(X, y, class_in_use = None, verbose = False):
    y = np.array(y) 
    if class_in_use is None:
        idx = np.ones_like(y, dtype = bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis = 0)
    X_incomplete, y_incomplete = X[idx], y[idx]
    if verbose == True:
        print("X shape :", X_incomplete.shape)
        print("y shape :", y_incomplete.shape)
    return X_incomplete, y_incomplete



def generate_imbal_CIFAR_private_data(X, y, y_super, classes_per_party, N_parties,
                                      samples_per_class=7):
    priv_data = [None] * N_parties
    combined_idxs = []
    count = 0
    y = np.array(y) 
    y_super = np.array(y_super) 
    for subcls_list in classes_per_party:
        idxs_per_party = []
        for c in subcls_list:
            idxs = np.flatnonzero(y == c)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            idxs_per_party.append(idxs)
        idxs_per_party = np.hstack(idxs_per_party)
        combined_idxs.append(idxs_per_party)
        
        dict_to_add = {}
        dict_to_add["idx"] = idxs_per_party
        dict_to_add["X"] = X[idxs_per_party]
        #dict_to_add["y"] = y[idxs_per_party]
        #dict_to_add["y_super"] = y_super[idxs_per_party]
        dict_to_add["y"] = y_super[idxs_per_party]
        priv_data[count] = dict_to_add
        count += 1
    
    combined_idxs = np.hstack(combined_idxs)
    total_priv_data = {}
    total_priv_data["idx"] = combined_idxs
    total_priv_data["X"] = X[combined_idxs]
    #total_priv_data["y"] = y[combined_idxs]
    #total_priv_data["y_super"] = y_super[combined_idxs]
    total_priv_data["y"] = y_super[combined_idxs]
    return priv_data, total_priv_data