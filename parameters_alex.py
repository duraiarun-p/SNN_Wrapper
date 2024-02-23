# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:15:51 2024

@author: pxb23215
"""
# %%
import argparse
# import torch
# import numpy as np
# %%
parser = argparse.ArgumentParser(description='Parameters for Alex model',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--reload', default=None, type=str, help='Path to weights to reload')
parser.add_argument('--fine_tune', default=False, action='store_true',
                    help='Does not reload conv1, FC and starts from epoch0')
parser.add_argument('--seed', default=0, type=int, help='Random seed')

parser.add_argument('--num_workers', default=1, type=int, help='Number of workers')

parser.add_argument('--lr', default=0.0268, type=float, help='Learning rate')
parser.add_argument('--leak_mem', default=0.874, type=float, help='Membrane leakage')
parser.add_argument('--arch', default='sresnet', type=str, help='[sresnet, sresnet_nm]')
parser.add_argument('--n', default=6, type=int, help='Depth scaling of the S-ResNet')
parser.add_argument('--nFilters', default=32, type=int, help='Width scaling of the S-ResNet')
parser.add_argument('--boosting', default=False, action='store_true', help='Use boosting layer')


parser.add_argument('--train_display_freq', default=1, type=int, help='Display_freq for train')
parser.add_argument('--test_display_freq', default=1, type=int, help='Display_freq for test')

parser.add_argument('--poisson_gen', default=False, action='store_true', help='Use poisson spike generation')

# Parsing command-line interface inputs
global args
args = parser.parse_args()
# Network setup parameters
