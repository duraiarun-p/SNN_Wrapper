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
    from wrapper_data_loader import custom_data_loader
    
    
    # from torchsummary import summary
    import torchinfo
    from parameters_alex import args as args_model
    
    from model_generate_alex import generate_nw_model
    
    # %%Parse input arguments
    parser = argparse.ArgumentParser(description='SNN Wrapper',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder', default='data', type=str, help='Folder for saving data')
    
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
    
    
    
    print('Using GPU device: ' + str(args_wrapper.device) + ' ' + str(torch.cuda.get_device_name()))
    device = torch.device("cuda:"+str(args_wrapper.device))
    
    # Model specific parameters
    model_summary_filename=args_model.mdl_summ_fname
    weight_folder=args_model.weight_folder
    num_workers=args_model.num_workers

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
    
    model=generate_nw_model(args_model,img_size,num_cls,num_steps)
    
    # Passing model to GPU
    model = model.to(device)
    # model_stats=summary(model,input_size=(3,32,32),col_names=["kernel_size","output_size"])
    # Summary using torchinfo library
    model_stat=torchinfo.summary(model, (3, 32, 32), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 0)
    summary_str=str(model_stat)
    # Writing into a text file : note encoding is required for this pkg
    with open(model_summary_filename, "w",encoding="utf-8") as text_file:
        text_file.write(summary_str)
    
        
        
    # %% Training operation (Scratch Training or Update-based Training)

    # %% Testing operation

    # %% Model saving

    # %% Visualisation

#%%
