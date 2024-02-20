import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import numpy as np
import os


def custom_data_loader(dataset, data_folder, num_steps):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        num_cls = 10
        img_size = 32

        train_set = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                                 download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=data_folder, train=False,
                                                download=True, transform=transform_test)
    elif dataset == 'cifar100':
        num_cls = 100
        img_size = 32

        train_set = torchvision.datasets.CIFAR100(root=data_folder, train=True,
                                                  download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root=data_folder, train=False,
                                                 download=True, transform=transform_test)
    elif dataset == 'cifar10dvs':
        num_cls = 10
        img_size = 64

        split_by = 'number'
        normalization = None
        T = num_steps  # number of frames

        dataset_dir = os.path.join(data_folder, dataset)
        if os.path.isdir(dataset_dir) is not True:
            os.mkdir(dataset_dir)

        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        # from spikingjelly.datasets import split_to_train_test_set  # Original function

        # Redefining split function to make it faster
        def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int,
                                    random_split: bool = False):
            '''
            :param train_ratio: split the ratio of the origin dataset as the train set
            :type train_ratio: float
            :param origin_dataset: the origin dataset
            :type origin_dataset: torch.utils.data.Dataset
            :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
            :type num_classes: int
            :param random_split: If ``False``, the front ratio of samples in each classes will
                    be included in train set, while the reset will be included in test set.
                    If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
                    ``numpy.randon.seed``
            :type random_split: int
            :return: a tuple ``(train_set, test_set)``
            :rtype: tuple
            '''
            import math
            label_idx = []

            if len(origin_dataset.samples) != 10000:  # If number of samples has been modified store label one by one
                for i in range(num_classes):
                    label_idx.append([])
                for i, item in enumerate(origin_dataset):
                    y = item[1]
                    if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
                        y = y.item()
                    label_idx[y].append(i)
            else:
                for i in range(10):  # Else, 1000 images per class
                    label_idx.append(list(range(i * 1000, (i + 1) * 1000)))
            train_idx = []
            test_idx = []
            if random_split:
                for i in range(num_classes):
                    np.random.shuffle(label_idx[i])

            for i in range(num_classes):
                pos = math.ceil(label_idx[i].__len__() * train_ratio)
                train_idx.extend(label_idx[i][0: pos])
                test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

            return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

        origin_set = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T,
                                split_by=split_by)
        train_set, test_set = split_to_train_test_set(0.9, origin_set, 10)

    else:
        print("Dataset name not found")
        exit()
    return train_set, test_set
