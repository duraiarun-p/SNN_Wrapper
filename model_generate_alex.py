# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:11:47 2024

@author: pxb23215
"""
# Model Authour: Alex Vincent
# Model Authour: https://github.com/VicenteAlex/Spiking_ResNet 

#%% Importing Libraries

import torch
import torchinfo
import numpy as np

from model_alex_sola import SResnet, SResnetNM

from parameters_alex import args as args_model
# from wrapper_main import args_wrapper

#%%
global args_model

def generate_nw_model(args_model,img_size,num_cls,num_steps):
    
    # Wrapper parameters
    # num_steps = args_wrapper.num_steps    
            
    
    # Model specific parameters
    weight_folder = args_model.weight_folder
    num_workers = args_model.num_workers
    arch = args_model.arch
    n = args_model.n
    nFilters = args_model.nFilters
    boosting = args_model.boosting
    poisson_gen = args_model.poisson_gen
    leak_mem = args_model.leak_mem    
    lr   = args_model.lr
    
    
    # Initialize random seed
    seed = args_model.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    
    #%%
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
        
    return model

