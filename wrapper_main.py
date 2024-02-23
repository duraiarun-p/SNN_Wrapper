# -*- coding: utf-8 -*-
# Python Script created using PyCharm IDE
# Created by pxb23215 at 20/02/2024
###############################################
if __name__ == '__main__':
    # %% Import Libraries
    import sys
    import os
    print('Python %s on %s' % (sys.version, sys.platform))
    import argparse
    import torch
    import numpy as np
    from wrapper_data_loader import custom_data_loader
    from model_alex_sola import SResnet, SResnetNM
    
    # from torchsummary import summary
    import torchinfo
    
    from parameters_alex import args as args_model
    
    # %%Parse input arguments
    parser = argparse.ArgumentParser(description='SNN Wrapper',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder', default='data', type=str, help='Folder for saving data')
    # Change for different network models
    parser.add_argument('--weight_folder', default='models_alex', type=str, help='Folder for saving weights')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
    parser.add_argument('--num_epochs', default=70, type=int, help='Number of epochs')    
    parser.add_argument('--num_steps', default=50, type=int, help='Number of time-step')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='[cifar10, cifar100, cifar10dvs]')
    parser.add_argument('--device', default='0', type=str, help='GPU to use')
    
    
    global args_wrapper
    args_wrapper = parser.parse_args()

    
    # Common for all models
    data_folder = args_wrapper.data_folder
    num_steps = args_wrapper.num_steps
    dataset = args_wrapper.dataset
    batch_size      = args_wrapper.batch_size
    batch_size_test = args_wrapper.batch_size*2
    num_epochs      = args_wrapper.num_epochs
    num_steps       = args_wrapper.num_steps
    
    # Model specific parameters
    weight_folder = args_wrapper.weight_folder
    num_workers = args_model.num_workers
    arch = args_model.arch
    n = args_model.n
    nFilters = args_model.nFilters
    boosting = args_model.boosting
    poisson_gen = args_model.poisson_gen
    leak_mem = args_model.leak_mem    
    lr   = args_model.lr
    
    print('Using GPU device: ' + str(args_wrapper.device) + ' ' + str(torch.cuda.get_device_name()))
    device = torch.device("cuda:"+str(args_wrapper.device))

    # Initialize random seed
    seed = args_model.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
   

    # %% Dataset downloader and Dataset object generator (Training and test set objects)
    
    # Define folder where weights are saved
    if os.path.isdir(weight_folder) is not True:
        os.mkdir(weight_folder)

    # Define folder where data is saved
    if os.path.isdir(data_folder) is not True:
        os.mkdir(data_folder)
    
    print(data_folder)
    train_set, test_set, img_size, num_cls = custom_data_loader(dataset, data_folder, num_steps)# Dataset objects
    # %% Dataloader (Training and test data)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    # %% Model specification and generation
    # Model setup parameters?
    # Set up network architecture and optimizer
    if arch == 'sresnet':
        model = SResnet(n=n, nFilters=nFilters, num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls,
                        boosting=boosting, poisson_gen=poisson_gen)
        print(arch,'model created')
        # summary(model,(img_size,img_size))
    elif arch == 'sresnet_nm':
        model = SResnetNM(n=n, nFilters=nFilters, num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls)
        print(arch,'model created')
    else:
        print("Architecture name not found")
        exit()
    
    # Passing model to GPU
    model = model.to(device)
    # model_stats=summary(model,input_size=(3,32,32),col_names=["kernel_size","output_size"])
    # Summary using torchinfo library
    model_stat=torchinfo.summary(model, (3, 32, 32), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 0)
    summary_str=str(model_stat)
    # Writing into a text file : note encoding is required for this pkg
    with open("model_summary.txt", "w",encoding="utf-8") as text_file:
        text_file.write(summary_str)
        
        
    # %% Training operation (Scratch Training or Update-based Training)

    # %% Testing operation

    # %% Model saving

    # %% Visualisation

#%%
