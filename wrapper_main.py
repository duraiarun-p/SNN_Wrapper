# -*- coding: utf-8 -*-
# Python Script created using PyCharm IDE
# Created by pxb23215 at 20/02/2024
###############################################
if __name__ == '__main__':
    # %% Import Libraries
    import sys

    print('Python %s on %s' % (sys.version, sys.platform))
    import argparse
    import torch
    import numpy as np
    from wrapper_data_loader import custom_data_loader
    from model_alex_sola import SResnet, SResnetNM
    # %%Parse input arguments
    parser = argparse.ArgumentParser(description='SNN Wrapper',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder', default='data', type=str, help='Folder for saving data')
    parser.add_argument('--weight_folder', default='models', type=str, help='Folder for saving weights')
    parser.add_argument('--reload', default=None, type=str, help='Path to weights to reload')
    parser.add_argument('--fine_tune', default=False, action='store_true',
                        help='Does not reload conv1, FC and starts from epoch0')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--num_steps', default=50, type=int, help='Number of time-step')
    parser.add_argument('--batch_size', default=21, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.0268, type=float, help='Learning rate')
    parser.add_argument('--leak_mem', default=0.874, type=float, help='Membrane leakage')
    parser.add_argument('--arch', default='sresnet', type=str, help='[sresnet, sresnet_nm]')
    parser.add_argument('--n', default=6, type=int, help='Depth scaling of the S-ResNet')
    parser.add_argument('--nFilters', default=32, type=int, help='Width scaling of the S-ResNet')
    parser.add_argument('--boosting', default=False, action='store_true', help='Use boosting layer')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='[cifar10, cifar100, cifar10dvs]')
    parser.add_argument('--num_epochs', default=70, type=int, help='Number of epochs')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--train_display_freq', default=1, type=int, help='Display_freq for train')
    parser.add_argument('--test_display_freq', default=1, type=int, help='Display_freq for test')
    parser.add_argument('--device', default='0', type=str, help='GPU to use')
    parser.add_argument('--poisson_gen', default=False, action='store_true', help='Use poisson spike generation')

    # Declaring
    global args
    args = parser.parse_args()
    # Network setup parameters
    data_folder = args.data_folder
    num_steps = args.num_steps
    dataset = args.dataset
    num_workers = args.num_workers
    
    n = args.n
    nFilters = args.nFilters
    boosting = args.boosting
    poisson_gen = args.poisson_gen
    leak_mem = args.leak_mem
    batch_size      = args.batch_size
    batch_size_test = args.batch_size*2
    num_epochs      = args.num_epochs
    num_steps       = args.num_steps
    lr   = args.lr
    
    # Initialize random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # %% Dataset downloader and Dataset object generator (Training and test set objects)
    
    print(data_folder)
    train_set, test_set, img_size, num_cls = custom_data_loader(dataset, data_folder, num_steps)# Dataset objects
    # %% Dataloader (Training and test data)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    # %% Model specification and generation
    # Set up network architecture and optimizer
    if args.arch == 'sresnet':
        model = SResnet(n=n, nFilters=nFilters, num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls,
                        boosting=boosting, poisson_gen=poisson_gen)
        # print(model)
        # summary(model,(img_size,img_size))
    elif args.arch == 'sresnet_nm':
        model = SResnetNM(n=n, nFilters=nFilters, num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls)

    else:
        print("Architecture name not found")
        exit()

    # %% Training operation (Scratch Training or Update-based Training)

    # %% Testing operation

    # %% Model saving

    # %% Visualisation

