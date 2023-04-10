'''Modified from https://github.com/alinlab/LfF/blob/master/data/util.py'''


import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import Sampler, random_split
from torchvision import transforms as T
from data.attr_dataset import AttributeDataset
from functools import reduce


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class IdxDataset2(Dataset):
    def __init__(self, dataset):
        self.full_dataset = dataset
        train_set_size = int(len(self.full_dataset) * 0.9)
        valid_set_size = len(self.full_dataset) - train_set_size
        self.train_set, self.valid_set = random_split(self.full_dataset, [train_set_size, valid_set_size])
        self.dataset = self.train_set
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])

    def make_train(self):
        self.dataset = self.train_set

    def make_biased_val(self):
        self.dataset = self.valid_set

    def make_fulltrain(self):
        self.dataset = self.full_dataset

class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item

    
transforms = {
    "ColoredMNIST": {
        "train": T.Compose([T.ToTensor()]),
        "eval": T.Compose([T.ToTensor()])
    },
    "CorruptedCIFAR10": {
        "train": T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(32,scale=(0.5, 1.0)), #Scale of randomcrop+padding=4 would equal 0.765625
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "eval": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
    "Camelyon17": {
        "train": T.Compose(
            [
                T.ToPILImage(),
                T.CenterCrop(32),
                T.RandomResizedCrop(32,scale=(0.5, 1.0)), #Scale of randomcrop+padding=4 would equal 0.765625
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        ),
        "eval": T.Compose(
            [
                T.ToPILImage(),
                T.CenterCrop(32),
                T.ToTensor(),
            ]
        ),
    },


}

def get_dataset_tag(config):
    bias_confl_perc = config["data"]["bias_conflicting_perc"] 
    severity = config["data"]["severity"]
    dataset = config["data"]["dataset"]
    if dataset == "colored_mnist":
        dataset_tag = f"ColoredMNIST-Skewed{bias_confl_perc}-Severity{severity}"
    elif dataset == "cifar10_type0":
        dataset_tag = f"CorruptedCIFAR10-Type0-Skewed{bias_confl_perc}-Severity{severity}"
    elif dataset == "cifar10_type1":
        dataset_tag = f"CorruptedCIFAR10-Type1-Skewed{bias_confl_perc}-Severity{severity}"   
    elif dataset == "camelyon17_type0":
        dataset_tag = f"Camelyon17-Type0-Skewed{bias_confl_perc}"
    elif dataset == "camelyon17_type1":
        dataset_tag = f"Camelyon17-Type1-Skewed{bias_confl_perc}"
    elif dataset == "camelyon17_type2":
        dataset_tag = f"Camelyon17-Type2-Skewed{bias_confl_perc}"
    else:
        raise NotImplementedError("Dataset not implemented.")
    return dataset_tag

def get_dataset(config, dataset_split):
    dataset_tag = get_dataset_tag(config)
    dataset_category = dataset_tag.split("-")[0]
    data_dir = config["user"]["data_dir"]
    root = os.path.join(data_dir, dataset_tag)
    transform = transforms[dataset_category][dataset_split]
    dataset_split = "test" if (dataset_split == "eval") else dataset_split
    dataset = AttributeDataset(
        root=root, split=dataset_split, transform=transform
    )

    return dataset


