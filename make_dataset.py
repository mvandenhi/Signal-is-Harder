'''Modified from https://github.com/alinlab/LfF/blob/master/util.py'''

import os
from re import A
from xmlrpc.client import Boolean
from tqdm import tqdm
import pickle

import numpy as np
import torch
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as T

from data.corrupted_cifar10_protocol import CORRUPTED_CIFAR10_PROTOCOL
from data.colored_mnist_protocol import COLORED_MNIST_PROTOCOL
from data.rotated_mnist_protocol import ROTATED_MNIST_PROTOCOL
from data.shifted_mnist_protocol import SHIFTED_MNIST_PROTOCOL


import yaml 
import argparse

from util import set_seed

def make_attr_labels(target_labels, bias_aligned_ratio):
    num_classes = target_labels.max().item() + 1
    num_samples_per_class = np.array(
        [
            torch.sum(target_labels == label).item()
            for label in range(num_classes)
        ]
    )
    ratios_per_class = bias_aligned_ratio * np.eye(num_classes) + (
        1 - bias_aligned_ratio
    ) / (num_classes - 1) * (1 - np.eye(num_classes))

    corruption_milestones_per_class = (
        num_samples_per_class[:, np.newaxis]
        * np.cumsum(ratios_per_class, axis=1)
    ).round()

    attr_labels = torch.zeros_like(target_labels)
    for label in range(num_classes):
        indices = (target_labels == label).nonzero().squeeze()
        corruption_milestones = corruption_milestones_per_class[label]
        for corruption_idx, idx in enumerate(indices):
            attr_labels[idx] = np.min(
                np.nonzero(corruption_milestones > corruption_idx)[0]
            ).item()

    return attr_labels

def make_corrupted_cifar10(
    data_dir, skewed_ratio, corruption_names, severity, config, postfix="0"
    ):
    cifar10_dir = os.path.join(data_dir, "CIFAR10")
    corrupted_cifar10_dir = os.path.join(
        data_dir, f"CorruptedCIFAR10-Type{postfix}-Skewed{skewed_ratio}-Severity{severity}"
    )
    os.makedirs(corrupted_cifar10_dir, exist_ok=True)
    print(corrupted_cifar10_dir)
    protocol = CORRUPTED_CIFAR10_PROTOCOL
    convert_img = T.Compose([T.ToTensor(), T.ToPILImage()])

    attr_names = ["object", "corruption"]
    attr_names_path = os.path.join(corrupted_cifar10_dir, "attr_names.pkl")
    with open(attr_names_path, "wb") as f:
        pickle.dump(attr_names, f)

    for split in ["train", "test"]:
        dataset = CIFAR10(cifar10_dir, train=(split == "train"), download=True)
        os.makedirs(os.path.join(corrupted_cifar10_dir, split), exist_ok=True)

        if split == "train":
            bias_aligned_ratio = 1-skewed_ratio
        else:
            bias_aligned_ratio = 0.1

        corruption_labels = make_attr_labels(
            torch.LongTensor(dataset.targets), bias_aligned_ratio
        )

        images, attrs = [], []
        for img, target_label, corruption_label in tqdm(
            zip(dataset.data, dataset.targets, corruption_labels),
            total=len(corruption_labels),
        ):
            
            method_name = corruption_names[corruption_label]
            corrupted_img = protocol[method_name](convert_img(img), severity+1)
            images.append(np.array(corrupted_img).astype(np.uint8))
            attrs.append([target_label, corruption_label])
                    

        # For Dfa reproducibility: Separately save data as they expect it. Careful this is hardcoded! Only uses Severity4!
        if config["Dfa"]["dataset"] and severity == 4: 
            import imageio
            from distutils.dir_util import copy_tree            
            dfa_ratio = skewed_ratio * 100
            if postfix == '0':
                path = config["Dfa"]["data_dir"]+f'/cifar10c/{dfa_ratio:g}pct'
            elif postfix == '1': path = config["Dfa"]["data_dir"]+f'/cifar10ct1/{dfa_ratio:g}pct'
            else: raise NotImplementedError
            attr = np.array(attrs)
            imgs = np.array(images)
            if split == "train":
                for j in range(len(np.unique(attr))):
                    ind = np.nonzero((attr[:,0] == j) & (attr[:,0] == attr[:,1]))[0]
                    os.makedirs(os.path.join(path, "align", f'{j}'), exist_ok=True)
                    os.makedirs(os.path.join(path, "conflict", f'{j}'), exist_ok=True)
                    for i in range(len(ind)):
                        path_img = os.path.join(path, "align", f'{attr[ind[0]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
                        imageio.imwrite(path_img, imgs[ind[i],:,:,:])
                    ind = np.nonzero((attr[:,0] == j) & (attr[:,0] != attr[:,1]))[0]
                    for i in range(len(ind)):
                        path_img = os.path.join(path, "conflict", f'{attr[ind[0]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
                        imageio.imwrite(path_img, imgs[ind[i],:,:,:])
            elif split == "test": 
                for j in range(len(np.unique(attr))):
                    os.makedirs(os.path.join(path, "test", f'{j}'), exist_ok=True)
                for i in range(len(attr[:,0])):
                    path_img = os.path.join(path, "test", f'{attr[i,0]}', f"{i}_{attr[i,0]}_{attr[i,1]}.png")
                    imageio.imwrite(path_img, imgs[i,:,:,:])
                #Create Pseudovalidation set as it's never used
                os.makedirs(os.path.join(path, "valid"), exist_ok=True)
                copy_tree(os.path.join(path, "test"), os.path.join(path, "valid"))
            
            else: raise NotImplementedError



        image_path = os.path.join(corrupted_cifar10_dir, split, "images.npy")
        np.save(image_path, np.array(images).astype(np.uint8))
        attr_path = os.path.join(corrupted_cifar10_dir, split, "attrs.npy")
        np.save(attr_path, np.array(attrs).astype(np.uint8))

def make_colored_mnist(data_dir, skewed_ratio, severity, config):
    mnist_dir = os.path.join(data_dir, "MNIST")
    colored_mnist_dir = os.path.join(
        data_dir, f"ColoredMNIST-Skewed{skewed_ratio}-Severity{severity}"
    )
    os.makedirs(colored_mnist_dir, exist_ok=True)
    print(colored_mnist_dir)
    protocol = COLORED_MNIST_PROTOCOL
    attr_names = ["digit", "color"]
    attr_names_path = os.path.join(colored_mnist_dir, "attr_names.pkl")
    with open(attr_names_path, "wb") as f:
        pickle.dump(attr_names, f)

    for split in ["train", "test"]:
        dataset = MNIST(mnist_dir, train=(split == "train"), download=True)
        os.makedirs(os.path.join(colored_mnist_dir, split), exist_ok=True)

        if split == "train":
            bias_aligned_ratio = 1. - skewed_ratio
        else:
            bias_aligned_ratio = 0.1

        color_labels = make_attr_labels(
            torch.LongTensor(dataset.targets), bias_aligned_ratio
        )

        images, attrs = [], []
        for img, target_label, color_label in tqdm(
            zip(dataset.data, dataset.targets, color_labels),
            total=len(color_labels),
        ):
            colored_img = protocol[color_label.item()](img, severity)
            # Change RBG from first to last dimension
            colored_img = np.moveaxis(np.uint8(colored_img), 0, 2)

            images.append(colored_img)
            attrs.append([target_label, color_label])
        

        # For Dfa reproducibility: Separately save data as they expect it. Careful this is hardcoded! Only uses Severity4!
        if config["Dfa"]["dataset"] and severity == 4: 
            import imageio
            print("Creating dataset for Dfa too!")
            from distutils.dir_util import copy_tree            
            dfa_ratio = skewed_ratio * 100
            path = config["Dfa"]["data_dir"]+f'/cmnist/{dfa_ratio:g}pct'
            attr = np.array(attrs)
            imgs = np.array(images)
            if split == "train":
                for j in range(len(np.unique(attr))):
                    ind = np.nonzero((attr[:,0] == j) & (attr[:,0] == attr[:,1]))[0]
                    os.makedirs(os.path.join(path, "align", f'{j}'), exist_ok=True)
                    os.makedirs(os.path.join(path, "conflict", f'{j}'), exist_ok=True)
                    for i in range(len(ind)):
                        path_img = os.path.join(path, "align", f'{attr[ind[0]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
                        imageio.imwrite(path_img, imgs[ind[i],:,:,:])
                    ind = np.nonzero((attr[:,0] == j) & (attr[:,0] != attr[:,1]))[0]
                    for i in range(len(ind)):
                        path_img = os.path.join(path, "conflict", f'{attr[ind[0]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
                        imageio.imwrite(path_img, imgs[ind[i],:,:,:])
            elif split == "test": 
                for j in range(len(np.unique(attr))):
                    os.makedirs(os.path.join(path, "test", f'{j}'), exist_ok=True)
                for i in range(len(attr[:,0])):
                    path_img = os.path.join(path, "test", f'{attr[i,0]}', f"{i}_{attr[i,0]}_{attr[i,1]}.png")
                    imageio.imwrite(path_img, imgs[i,:,:,:])
                #Create Pseudovalidation set as it's never used
                os.makedirs(os.path.join(path, "valid"), exist_ok=True)
                copy_tree(os.path.join(path, "test"), os.path.join(path, "valid"))
            else: raise NotImplementedError



        image_path = os.path.join(colored_mnist_dir, split, "images.npy")
        np.save(image_path, np.array(images).astype(np.uint8))
        attr_path = os.path.join(colored_mnist_dir, split, "attrs.npy")
        np.save(attr_path, np.array(attrs).astype(np.uint8))

    
def make_rotated_mnist(data_dir, skewed_ratio, severity, config):
    mnist_dir = os.path.join(data_dir, "MNIST")
    rotated_mnist_dir = os.path.join(
        data_dir, f"RotatedMNIST-Skewed{skewed_ratio}-Severity{severity}"
    )
    os.makedirs(rotated_mnist_dir, exist_ok=True)
    print(rotated_mnist_dir)
    protocol = ROTATED_MNIST_PROTOCOL
    attr_names = ["digit", "rotation"]
    attr_names_path = os.path.join(rotated_mnist_dir, "attr_names.pkl")
    with open(attr_names_path, "wb") as f:
        pickle.dump(attr_names, f)

    for split in ["train", "test"]:
        dataset = MNIST(mnist_dir, train=(split == "train"), download=True)
        os.makedirs(os.path.join(rotated_mnist_dir, split), exist_ok=True)

        if split == "train":
            bias_aligned_ratio = 1. - skewed_ratio
        else:
            bias_aligned_ratio = 0.5
        #Keep only 3 and 8 and change their classes to 0,1
        targets = ((dataset.targets==3) | (dataset.targets==8)).nonzero()
        data = dataset.data[targets].squeeze(1)
        data_labels = dataset.targets[targets].squeeze(1)
        data_labels[(data_labels == 3).nonzero()] = 0
        data_labels[(data_labels == 8).nonzero()] = 1

        rotation_labels = make_attr_labels(
            torch.LongTensor(data_labels), bias_aligned_ratio
        )

        images, attrs = [], []
        for img, target_label, rotation_label in tqdm(
            zip(data, data_labels, rotation_labels),
            total=len(rotation_labels),
        ):
            rotated_img = protocol[rotation_label.item()](img, severity)

            images.append(rotated_img)
            attrs.append([target_label, rotation_label])

        # For Dfa reproducibility: Separately save data as they expect it. Careful this is hardcoded! Only uses Severity4!
        if config["Dfa"]["dataset"] and severity == 4: 
            import imageio
            from distutils.dir_util import copy_tree            
            dfa_ratio = skewed_ratio * 100
            path = config["Dfa"]["data_dir"]+f'/rmnist/{dfa_ratio:g}pct'
            attr = np.array(attrs)
            imgs = [np.array(image).astype(np.uint8) for image in images]
            if split == "train":
                for j in range(len(np.unique(attr))):
                    ind = np.nonzero((attr[:,0] == j) & (attr[:,0] == attr[:,1]))[0]
                    os.makedirs(os.path.join(path, "align", f'{j}'), exist_ok=True)
                    os.makedirs(os.path.join(path, "conflict", f'{j}'), exist_ok=True)
                    for i in range(len(ind)):
                        path_img = os.path.join(path, "align", f'{attr[ind[0]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
                        imageio.imwrite(path_img, imgs[ind[i]])
                    ind = np.nonzero((attr[:,0] == j) & (attr[:,0] != attr[:,1]))[0]
                    for i in range(len(ind)):
                        path_img = os.path.join(path, "conflict", f'{attr[ind[0]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
                        imageio.imwrite(path_img, imgs[ind[i]])
            elif split == "test": 
                for j in range(len(np.unique(attr))):
                    os.makedirs(os.path.join(path, "test", f'{j}'), exist_ok=True)
                for i in range(len(attr[:,0])):
                    path_img = os.path.join(path, "test", f'{attr[i,0]}', f"{i}_{attr[i,0]}_{attr[i,1]}.png")
                    imageio.imwrite(path_img, imgs[i])
                #Create Pseudovalidation set as it's never used
                os.makedirs(os.path.join(path, "valid"), exist_ok=True)
                copy_tree(os.path.join(path, "test"), os.path.join(path, "valid"))
            else: raise NotImplementedError

        image_path = os.path.join(rotated_mnist_dir, split, "images.npy")
        np.save(image_path, [np.array(image).astype(np.uint8) for image in images])
        attr_path = os.path.join(rotated_mnist_dir, split, "attrs.npy")
        np.save(attr_path, np.array(attrs).astype(np.uint8))



    
def make_shifted_mnist(data_dir, skewed_ratio, severity, config):
    mnist_dir = os.path.join(data_dir, "MNIST")
    shifted_mnist_dir = os.path.join(
        data_dir, f"ShiftedMNIST-Skewed{skewed_ratio}-Severity{severity}"
    )
    os.makedirs(shifted_mnist_dir, exist_ok=True)
    print(shifted_mnist_dir)
    protocol = SHIFTED_MNIST_PROTOCOL
    attr_names = ["digit", "rotation"]
    attr_names_path = os.path.join(shifted_mnist_dir, "attr_names.pkl")
    with open(attr_names_path, "wb") as f:
        pickle.dump(attr_names, f)

    for split in ["train", "test"]:
        dataset = MNIST(mnist_dir, train=(split == "train"), download=True)
        os.makedirs(os.path.join(shifted_mnist_dir, split), exist_ok=True)

        if split == "train":
            bias_aligned_ratio = 1. - skewed_ratio
        else:
            bias_aligned_ratio = 0.5
        #Keep only 3 and 8 and change their classes to 0,1
        targets = ((dataset.targets==3) | (dataset.targets==8)).nonzero()
        data = dataset.data[targets].squeeze(1)
        data_labels = dataset.targets[targets].squeeze(1)
        data_labels[(data_labels == 3).nonzero()] = 0
        data_labels[(data_labels == 8).nonzero()] = 1

        shifted_labels = make_attr_labels(
            torch.LongTensor(data_labels), bias_aligned_ratio
        )

        images, attrs = [], []
        for img, target_label, shifted_label in tqdm(
            zip(data, data_labels, shifted_labels),
            total=len(shifted_labels),
        ):
            shifted_img = protocol[shifted_label.item()](img, severity)

            images.append(shifted_img)
            attrs.append([target_label, shifted_label])

        # For Dfa reproducibility: Separately save data as they expect it. Careful this is hardcoded! Only uses Severity4!
        if config["Dfa"]["dataset"] and severity == 4: 
            import imageio
            from distutils.dir_util import copy_tree            
            dfa_ratio = skewed_ratio * 100
            path = config["Dfa"]["data_dir"]+f'/smnist/{dfa_ratio:g}pct'
            attr = np.array(attrs)
            imgs = [np.array(image).astype(np.uint8) for image in images]
            if split == "train":
                for j in range(len(np.unique(attr))):
                    ind = np.nonzero((attr[:,0] == j) & (attr[:,0] == attr[:,1]))[0]
                    os.makedirs(os.path.join(path, "align", f'{j}'), exist_ok=True)
                    os.makedirs(os.path.join(path, "conflict", f'{j}'), exist_ok=True)
                    for i in range(len(ind)):
                        path_img = os.path.join(path, "align", f'{attr[ind[0]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
                        imageio.imwrite(path_img, imgs[ind[i]])
                    ind = np.nonzero((attr[:,0] == j) & (attr[:,0] != attr[:,1]))[0]
                    for i in range(len(ind)):
                        path_img = os.path.join(path, "conflict", f'{attr[ind[0]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
                        imageio.imwrite(path_img, imgs[ind[i]])
            elif split == "test": 
                for j in range(len(np.unique(attr))):
                    os.makedirs(os.path.join(path, "test", f'{j}'), exist_ok=True)
                for i in range(len(attr[:,0])):
                    path_img = os.path.join(path, "test", f'{attr[i,0]}', f"{i}_{attr[i,0]}_{attr[i,1]}.png")
                    imageio.imwrite(path_img, imgs[i])
                #Create Pseudovalidation set as it's never used
                os.makedirs(os.path.join(path, "valid"), exist_ok=True)
                copy_tree(os.path.join(path, "test"), os.path.join(path, "valid"))
            else: raise NotImplementedError

        image_path = os.path.join(shifted_mnist_dir, split, "images.npy")
        np.save(image_path, [np.array(image).astype(np.uint8) for image in images])
        attr_path = os.path.join(shifted_mnist_dir, split, "attrs.npy")
        np.save(attr_path, np.array(attrs).astype(np.uint8))

def make_camelyon17_type0(data_dir,skewed_ratio,config): 
    # Type0: Using data from all 4 training hospitals in testset with 50-50 positive and negative ratio
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, ConcatDataset
    from torch.utils.data.dataset import Dataset


    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset="camelyon17", download=True, unlabeled=False, root_dir=data_dir)    
    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose([transforms.ToTensor()])
        )
    idval_data = dataset.get_subset(
        "id_val",
        transform=transforms.Compose([transforms.ToTensor()])
        )
    oodval_data = dataset.get_subset(
        "val",
        transform=transforms.Compose([transforms.ToTensor()])
        )

    full_train_data = ConcatDataset([train_data,idval_data,oodval_data]) # NOT test_data 
    data_loader = DataLoader(
        full_train_data,
        shuffle=True,
        batch_size=1) # By shuffle all inputs from all datasets get randomly shuffled

    pos = np.zeros(2)
    while len(np.unique(pos))==1:
        pos = np.random.randint(0,4,2)
    pos[pos>=2]+=1 # 2 is test hospital
    #bias_label = np.zeros(4)
    #bias_label[pos] += 1
    #assert np.median(bias_label)==0.5

    camelyon_dir = os.path.join(
        data_dir, f"Camelyon17-Type0-Skewed{skewed_ratio}"
    )
    os.makedirs(camelyon_dir, exist_ok=True)
    print(camelyon_dir)
    attr_names = ["tumor", "hospital"]
    attr_names_path = os.path.join(camelyon_dir, "attr_names.pkl")
    with open(attr_names_path, "wb") as f:
        pickle.dump(attr_names, f)
    for split in ["train", "test"]:
        os.makedirs(os.path.join(camelyon_dir, split), exist_ok=True)
    test_images, test_attrs = [], []
    images, attrs = [], []
    bias_aligned_ratio = 1-skewed_ratio
    test_count = np.zeros((5,2)) # Count images in testset of all combinations
    for idx, (x, y, metadata) in enumerate(tqdm(data_loader)):
        if test_count[metadata[:,0].item(),y.item()]<1250: #10'000 testset images
            x = np.moveaxis(np.array(x), 1, 3)
            x *= 255
            test_images.append(np.array(x.squeeze(0)).astype(np.uint8))
            test_attrs.append([y.squeeze(), metadata[:,0].squeeze()])
            test_count[metadata[:,0].item(),y.item()]+=1
        else:
            include_align = np.random.binomial(1,bias_aligned_ratio,size=len(x))
            include_confl = 1-include_align
            pos_domains = np.isin(metadata[:,0],pos)
            aligned = np.zeros_like(include_align)
            aligned[pos_domains] = (y == 1)[pos_domains]
            aligned[~pos_domains] = (y == 0)[~pos_domains]
            aligned = aligned.astype(bool)
            include_imgs = np.zeros_like(include_align)
            include_imgs[aligned] = include_align[aligned]
            include_imgs[~aligned] = include_confl[~aligned]
            include_imgs = include_imgs.astype(bool)
            if include_imgs==0:
                continue
            x = np.moveaxis(np.array(x), 1, 3)
            x *= 255
            images.append(np.array(x[include_imgs].squeeze(0)).astype(np.uint8))
            attrs.append([y[include_imgs].squeeze(), metadata[:,0][include_imgs].squeeze()])
    assert ((test_count==0) | (test_count==1250)).all()

    # For Dfa reproducibility: Separately save data as they expect it. Careful this is hardcoded! Only uses Severity4!
    if config["Dfa"]["dataset"]:
        import imageio
        from distutils.dir_util import copy_tree            
        dfa_ratio = skewed_ratio * 100
        path = config["Dfa"]["data_dir"]+f'/camelyon17_type0/{dfa_ratio:g}pct'
        attr = np.array(attrs)
        imgs = [image for image in images]
        for j in range(2):
            os.makedirs(os.path.join(path, "align", f'{j}'), exist_ok=True)
            os.makedirs(os.path.join(path, "conflict", f'{j}'), exist_ok=True)
        pos_domains = np.isin(attr[:,1],pos)
        ind = np.zeros_like(attr[:,1])
        ind[pos_domains] = (attr[:,0] == 1)[pos_domains]
        ind[~pos_domains] = (attr[:,0] == 0)[~pos_domains]
        ind = np.nonzero(ind)[0]
        for i in tqdm(range(len(ind))):
            path_img = os.path.join(path, "align", f'{attr[ind[i]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
            imageio.imwrite(path_img, imgs[ind[i]])
        ind = np.zeros_like(attr[:,1])
        ind[pos_domains] = (attr[:,0] != 1)[pos_domains]
        ind[~pos_domains] = (attr[:,0] != 0)[~pos_domains]
        ind = np.nonzero(ind)[0]
        for i in tqdm(range(len(ind))):
            path_img = os.path.join(path, "conflict", f'{attr[ind[i]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
            imageio.imwrite(path_img, imgs[ind[i]])
        #Testset
        test_attr = np.array(test_attrs)
        test_imgs = [image for image in test_images]
        for j in range(2):
            os.makedirs(os.path.join(path, "test", f'{j}'), exist_ok=True)
        for i in tqdm(range(len(test_attr[:,0]))):
            path_img = os.path.join(path, "test", f'{test_attr[i,0]}', f"{i}_{test_attr[i,0]}_{test_attr[i,1]}.png")
            imageio.imwrite(path_img, test_imgs[i])
        #Create Pseudovalidation set as it's never used
        os.makedirs(os.path.join(path, "valid"), exist_ok=True)
        copy_tree(os.path.join(path, "test"), os.path.join(path, "valid"))

    image_path = os.path.join(camelyon_dir, "train", "images.npy")
    np.save(image_path, [np.array(image).astype(np.uint8) for image in images])
    attr_path = os.path.join(camelyon_dir, "train", "attrs.npy")
    np.save(attr_path, np.array(attrs).astype(np.uint8))

    image_path = os.path.join(camelyon_dir, "test", "images.npy")
    np.save(image_path, [np.array(image).astype(np.uint8) for image in test_images])
    attr_path = os.path.join(camelyon_dir, "test", "attrs.npy")
    np.save(attr_path, np.array(test_attrs).astype(np.uint8))

    


def make_camelyon17_type1(data_dir,skewed_ratio, config): 
    # Type1: Using data hospital 5 as testset as in original wilds
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, ConcatDataset
    from torch.utils.data.dataset import Dataset


    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset="camelyon17", download=True, unlabeled=False, root_dir=data_dir)    
    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose([transforms.ToTensor()])
        )
    idval_data = dataset.get_subset(
        "id_val",
        transform=transforms.Compose([transforms.ToTensor()])
        )
    oodval_data = dataset.get_subset(
        "val",
        transform=transforms.Compose([transforms.ToTensor()])
        )
    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose([transforms.ToTensor()])
        )
    full_train_data = ConcatDataset([train_data,idval_data,oodval_data]) # NOT test_data bcs this stays testset
    data_loader = DataLoader(
        full_train_data,
        shuffle=True,
        batch_size=1) # By shuffle all inputs from all datasets get randomly shuffled
    test_loader = DataLoader(
        test_data,
        shuffle=True,
        batch_size=1)

    pos = np.zeros(2)
    while len(np.unique(pos))==1:
        pos = np.random.randint(0,4,2)
    pos[pos>=2]+=1 # 2 is test hospital
    #bias_label = np.zeros(4)
    #bias_label[pos] += 1
    #assert np.median(bias_label)==0.5

    camelyon_dir = os.path.join(
        data_dir, f"Camelyon17-Type1-Skewed{skewed_ratio}"
    )
    os.makedirs(camelyon_dir, exist_ok=True)
    print(camelyon_dir)
    attr_names = ["tumor", "hospital"]
    attr_names_path = os.path.join(camelyon_dir, "attr_names.pkl")
    with open(attr_names_path, "wb") as f:
        pickle.dump(attr_names, f)

    for split in ["train", "test"]:
        os.makedirs(os.path.join(camelyon_dir, split), exist_ok=True)
        bias_aligned_ratio = 1-skewed_ratio
        if split == "train":
            images, attrs = [], []
            for idx, (x, y, metadata) in enumerate(tqdm(data_loader)):
                include_align = np.random.binomial(1,bias_aligned_ratio,size=len(x))
                include_confl = 1-include_align
                pos_domains = np.isin(metadata[:,0],pos)
                aligned = np.zeros_like(include_align)
                aligned[pos_domains] = (y == 1)[pos_domains]
                aligned[~pos_domains] = (y == 0)[~pos_domains]
                aligned = aligned.astype(bool)
                include_imgs = np.zeros_like(include_align)
                include_imgs[aligned] = include_align[aligned]
                include_imgs[~aligned] = include_confl[~aligned]
                include_imgs = include_imgs.astype(bool)
                if include_imgs==0:
                    continue
                x = np.moveaxis(np.array(x), 1, 3)
                x *= 255
                images.append(np.array(x[include_imgs].squeeze(0)).astype(np.uint8))
                attrs.append([y[include_imgs].squeeze(), metadata[:,0][include_imgs].squeeze()])
        else: 
            images, attrs = [], []
            for idx, (x, y, metadata) in enumerate(tqdm(test_loader)):
                x = np.moveaxis(np.array(x), 1, 3)
                x *= 255
                images.append(np.array(x.squeeze(0)).astype(np.uint8))
                attrs.append([y.squeeze(), metadata[:,0].squeeze()])
        

        # For Dfa reproducibility: Separately save data as they expect it. Careful this is hardcoded! Only uses Severity4!
        if config["Dfa"]["dataset"]:
            import imageio
            from distutils.dir_util import copy_tree            
            dfa_ratio = skewed_ratio * 100
            path = config["Dfa"]["data_dir"]+f'/camelyon17_type1/{dfa_ratio:g}pct'
            attr = np.array(attrs)
            imgs = [image for image in images]
            if split == "train":
                for j in range(2):
                    os.makedirs(os.path.join(path, "align", f'{j}'), exist_ok=True)
                    os.makedirs(os.path.join(path, "conflict", f'{j}'), exist_ok=True)
                pos_domains = np.isin(attr[:,1],pos)
                ind = np.zeros_like(attr[:,1])
                ind[pos_domains] = (attr[:,0] == 1)[pos_domains]
                ind[~pos_domains] = (attr[:,0] == 0)[~pos_domains]
                ind = np.nonzero(ind)[0]
                for i in tqdm(range(len(ind))):
                    path_img = os.path.join(path, "align", f'{attr[ind[i]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
                    imageio.imwrite(path_img, imgs[ind[i]])
                ind = np.zeros_like(attr[:,1])
                ind[pos_domains] = (attr[:,0] != 1)[pos_domains]
                ind[~pos_domains] = (attr[:,0] != 0)[~pos_domains]
                ind = np.nonzero(ind)[0]
                for i in tqdm(range(len(ind))):
                    path_img = os.path.join(path, "conflict", f'{attr[ind[i]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
                    imageio.imwrite(path_img, imgs[ind[i]])
            elif split == "test": 
                for j in range(2):
                    os.makedirs(os.path.join(path, "test", f'{j}'), exist_ok=True)
                for i in tqdm(range(len(attr[:,0]))):
                    path_img = os.path.join(path, "test", f'{attr[i,0]}', f"{i}_{attr[i,0]}_{attr[i,1]}.png")
                    imageio.imwrite(path_img, imgs[i])
                #Create Pseudovalidation set as it's never used
                os.makedirs(os.path.join(path, "valid"), exist_ok=True)
                copy_tree(os.path.join(path, "test"), os.path.join(path, "valid"))
            else: raise NotImplementedError

        image_path = os.path.join(camelyon_dir, split, "images.npy")
        np.save(image_path, [np.array(image).astype(np.uint8) for image in images])
        attr_path = os.path.join(camelyon_dir, split, "attrs.npy")
        np.save(attr_path, np.array(attrs).astype(np.uint8))

def make_camelyon17_type2(data_dir,skewed_ratio,config): 
    # Type2: Same as type0 but using first and testset hospital. hospital 1 is mostly positive, hospital 0 is mostly negative
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, ConcatDataset
    from torch.utils.data.dataset import Dataset


    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset="camelyon17", download=True, unlabeled=False, root_dir=data_dir)    
    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose([transforms.ToTensor()])
        )
    idval_data = dataset.get_subset(
        "id_val",
        transform=transforms.Compose([transforms.ToTensor()])
        )
    oodval_data = dataset.get_subset(
        "val",
        transform=transforms.Compose([transforms.ToTensor()])
        )
    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose([transforms.ToTensor()])
        )

    full_train_data = ConcatDataset([train_data,idval_data,oodval_data,test_data])
    data_loader = DataLoader(
        full_train_data,
        shuffle=True,
        batch_size=1) # By shuffle all inputs from all datasets get randomly shuffled

    pos = np.array([1])
    camelyon_dir = os.path.join(
        data_dir, f"Camelyon17-Type2-Skewed{skewed_ratio}"
    )
    os.makedirs(camelyon_dir, exist_ok=True)
    print(camelyon_dir)
    attr_names = ["tumor", "hospital"]
    attr_names_path = os.path.join(camelyon_dir, "attr_names.pkl")
    with open(attr_names_path, "wb") as f:
        pickle.dump(attr_names, f)
    for split in ["train", "test"]:
        os.makedirs(os.path.join(camelyon_dir, split), exist_ok=True)
    test_images, test_attrs = [], []
    images, attrs = [], []
    bias_aligned_ratio = 1-skewed_ratio
    test_count = np.zeros((5,2)) # Count images in testset of all combinations
    for idx, (x, y, metadata) in enumerate(tqdm(data_loader)): 
        # if not (metadata[:,0].item() in [0,2]): continue
        # elif test_count[metadata[:,0].item(),y.item()]<1250: # 5'000 testset images
        #     x = np.moveaxis(np.array(x), 1, 3)
        #     x *= 255
        #     test_count[metadata[:,0].item(),y.item()]+=1
        #     if metadata[:,0].squeeze() == 2: # Changing bias label s.t. it's 0&1. Only for this setting!
        #         metadata[:,0] = 1
        #     test_images.append(np.array(x.squeeze(0)).astype(np.uint8))
        #     test_attrs.append([y.squeeze(), metadata[:,0].squeeze()])
        if test_count[metadata[:,0].item(),y.item()]<1250 and (not metadata[:,0].item() in [0,2]): # 7'500 testset images
            x = np.moveaxis(np.array(x), 1, 3)
            x *= 255
            test_count[metadata[:,0].item(),y.item()]+=1
            test_images.append(np.array(x.squeeze(0)).astype(np.uint8))
            test_attrs.append([y.squeeze(), metadata[:,0].squeeze()])
        elif (not metadata[:,0].item() in [0,2]): continue
        else:
            if metadata[:,0].squeeze() == 2: # Changing bias label s.t. it's 0&1. Only for this setting!
                metadata[:,0] = 1
            include_align = np.random.binomial(1,bias_aligned_ratio,size=len(x))
            include_confl = 1-include_align
            pos_domains = np.isin(metadata[:,0],pos)
            aligned = np.zeros_like(include_align)
            aligned[pos_domains] = (y == 1)[pos_domains]
            aligned[~pos_domains] = (y == 0)[~pos_domains]
            aligned = aligned.astype(bool)
            include_imgs = np.zeros_like(include_align)
            include_imgs[aligned] = include_align[aligned]
            include_imgs[~aligned] = include_confl[~aligned]
            include_imgs = include_imgs.astype(bool)
            if include_imgs==0:
                continue
            x = np.moveaxis(np.array(x), 1, 3)
            x *= 255
            images.append(np.array(x[include_imgs].squeeze(0)).astype(np.uint8))
            attrs.append([y[include_imgs].squeeze(), metadata[:,0][include_imgs].squeeze()])
    assert ((test_count==0) | (test_count==1250)).all()

    # For Dfa reproducibility: Separately save data as they expect it. Careful this is hardcoded! Only uses Severity4!
    if config["Dfa"]["dataset"]:
        import imageio
        from distutils.dir_util import copy_tree            
        dfa_ratio = skewed_ratio * 100
        path = config["Dfa"]["data_dir"]+f'/camelyon17_type2/{dfa_ratio:g}pct'
        attr = np.array(attrs)
        imgs = [image for image in images]
        for j in range(2):
            os.makedirs(os.path.join(path, "align", f'{j}'), exist_ok=True)
            for f in os.listdir(os.path.join(path, "align", f'{j}')): # Remove already existing files
                os.remove(os.path.join(os.path.join(path, "align", f'{j}'), f))
            os.makedirs(os.path.join(path, "conflict", f'{j}'), exist_ok=True)
            for f in os.listdir(os.path.join(path, "conflict", f'{j}')): # Remove already existing files
                os.remove(os.path.join(os.path.join(path, "conflict", f'{j}'), f))
        pos_domains = np.isin(attr[:,1],pos)
        ind = np.zeros_like(attr[:,1])
        ind[pos_domains] = (attr[:,0] == 1)[pos_domains]
        ind[~pos_domains] = (attr[:,0] == 0)[~pos_domains]
        ind = np.nonzero(ind)[0]
        for i in tqdm(range(len(ind))):
            path_img = os.path.join(path, "align", f'{attr[ind[i]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
            imageio.imwrite(path_img, imgs[ind[i]])
        ind = np.zeros_like(attr[:,1])
        ind[pos_domains] = (attr[:,0] != 1)[pos_domains]
        ind[~pos_domains] = (attr[:,0] != 0)[~pos_domains]
        ind = np.nonzero(ind)[0]
        for i in tqdm(range(len(ind))):
            path_img = os.path.join(path, "conflict", f'{attr[ind[i]][0]}', f"{i}_{attr[ind[i],0]}_{attr[ind[i],1]}.png")
            imageio.imwrite(path_img, imgs[ind[i]])
        #Testset
        test_attr = np.array(test_attrs)
        test_imgs = [image for image in test_images]
        for j in range(2):
            os.makedirs(os.path.join(path, "test", f'{j}'), exist_ok=True)
            for f in os.listdir(os.path.join(path, "test", f'{j}')): # Remove already existing files
                os.remove(os.path.join(os.path.join(path, "test", f'{j}'), f))
        for i in tqdm(range(len(test_attr[:,0]))):
            path_img = os.path.join(path, "test", f'{test_attr[i,0]}', f"{i}_{test_attr[i,0]}_{test_attr[i,1]}.png")
            imageio.imwrite(path_img, test_imgs[i])
        #Create Pseudovalidation set as it's never used
        os.makedirs(os.path.join(path, "valid"), exist_ok=True)
        copy_tree(os.path.join(path, "test"), os.path.join(path, "valid"))

    image_path = os.path.join(camelyon_dir, "train", "images.npy")
    np.save(image_path, [np.array(image).astype(np.uint8) for image in images])
    attr_path = os.path.join(camelyon_dir, "train", "attrs.npy")
    np.save(attr_path, np.array(attrs).astype(np.uint8))

    image_path = os.path.join(camelyon_dir, "test", "images.npy")
    np.save(image_path, [np.array(image).astype(np.uint8) for image in test_images])
    attr_path = os.path.join(camelyon_dir, "test", "attrs.npy")
    np.save(attr_path, np.array(test_attrs).astype(np.uint8))



def make(make_target):

        # configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # manual overwriting of configuration for scripts
    #   initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help = "Choice of dataset")
    parser.add_argument("--random_state", default=None, type=int, help = "Random Seed")
    parser.add_argument("--Dfa_dataset", default=None, type=bool, help = "Create dataset for Dfa too?")
    args = parser.parse_args()
    #   replace all specified arguments
    updateable = [config["data"]["dataset"],config["random_state"],config["Dfa"]["dataset"]]
    values = []
    for i,v in enumerate(vars(args).values()):
        if v != None:
             values.append(v)
             print("Overwriting configuration")
        else: values.append(updateable[i])
    config["data"]["dataset"], config["random_state"], config["Dfa"]["dataset"] = values

    if make_target == None:
        make_target = config["data"]["dataset"]
    data_dir = config["user"]["data_dir"]
    random_state = config["random_state"]
    
    # Reproducibility
    set_seed(random_state)


    for skewed_ratio in [2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3]:

        #for severity in [1, 2, 3, 4]:                  This if from LfF but we only look at severity 4 here!
        for severity in [4]:
            if make_target == "colored_mnist":
                make_colored_mnist(data_dir=data_dir, skewed_ratio=skewed_ratio, severity=severity, config=config)
    
            if make_target == "cifar10_type0":
                make_corrupted_cifar10(
                    data_dir=data_dir,
                    corruption_names=[
                        "Snow",
                        "Frost",
                        "Fog",
                        "Brightness",
                        "Contrast",
                        "Spatter",
                        "Elastic",
                        "JPEG",
                        "Pixelate", 
                        "Saturate",
                    ],
                    skewed_ratio=skewed_ratio,
                    severity=severity,
                    config=config,
                    postfix="0"
                )
            
            if make_target == "cifar10_type1":            
                make_corrupted_cifar10(
                    data_dir=data_dir,
                    corruption_names=[
                        "Gaussian Noise",
                        "Shot Noise",
                        "Impulse Noise",
                        "Speckle Noise",
                        "Gaussian Blur",
                        "Defocus Blur",
                        "Glass Blur",
                        "Motion Blur",
                        "Zoom Blur",
                        "Original",
                    ],
                    skewed_ratio=skewed_ratio,
                    severity=severity,
                    config=config,
                    postfix="1"
                )
            if make_target == "rotated_mnist":
                make_rotated_mnist(data_dir=data_dir, skewed_ratio=skewed_ratio, severity=severity, config=config)
            if make_target == "shifted_mnist":
                make_shifted_mnist(data_dir=data_dir, skewed_ratio=skewed_ratio, severity=severity, config=config)
            if make_target == "camelyon17_type0":
                make_camelyon17_type0(data_dir=data_dir, skewed_ratio=skewed_ratio, config=config)
            if make_target == "camelyon17_type1":
                make_camelyon17_type1(data_dir=data_dir, skewed_ratio=skewed_ratio, config=config)
            if make_target == "camelyon17_type2":
                make_camelyon17_type2(data_dir=data_dir, skewed_ratio=skewed_ratio, config=config)
            

make(make_target=None)