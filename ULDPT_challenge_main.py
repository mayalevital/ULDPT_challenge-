#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:21:36 2022

@author: maya.fi@bm.technion.ac.il
"""


import os
from utilities import un_norm, train_test_net, train_val_test_por, ModelParamsInit, ModelParamsInit_unetr
import torch
import numpy as np
import matplotlib.pyplot as plt
from params import scan_params
import torch.nn as nn
import pandas as pd
from utilities import get_slice_ready, load_model, laplacian_filter, save_nift, results_summary, get_mat_compare
from utilities import arrange_data_old, pack_dcm
from unet_2 import BasicUNet
from dataset import ULDPT
import nibabel as nib
plt.ioff()
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

            
def trainloaders(params, data):
    _dataset = ULDPT(data)
    
    train_por, val_por = train_val_test_por(params, data)
    print("train portion size = ", len(train_por))
    print("test portion size = ", len(val_por))
    
    train_set = torch.utils.data.Subset(_dataset, train_por)
    val_set = torch.utils.data.Subset(_dataset, val_por)
       
    trainloader_1 = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'],
                                                shuffle=True, num_workers=4)
    trainloader_2 = torch.utils.data.DataLoader(val_set, batch_size=params['batch_size'],
                                                shuffle=True, num_workers=4)

    return trainloader_1, trainloader_2

CUDA_VISIBLE_DEVICES=1 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
params = scan_params()

t=params['t']
N = params['num_of_epochs']
l = params['lambda']
N_finish = params['N_finish']
root = '/tcmldrive/users/Maya/' #define your root dir

if(t==0): #arrange dataframe with the paths of the LD and ND DICOM pairs. root_dir is the path of the DICOM files for the uExplorer data
    params = scan_params()
    root_dir = '/tcmldrive/databases/Public/uExplorer'
    
    df = arrange_data_old(params, root_dir)
    df.to_pickle(os.path.join(root, 'all_doses_data_test.pkl'))

if(t==1): #train_test
    data = pd.read_pickle("./all_doses_data_test.pkl", compression='infer') #example
    opt = params['optimizer'][0]
    for wd in params['weight_decay']:
        for network in params['net']:
            for l in params['lambda']:
                for learn in params['lr']:
                    for method in params['method']:
                        print(network)
                        print('learning rate = ', learn) 
                        print('grads lambda = ', l)
                        print('optimizer ', opt)
                        print('N finish = ', N_finish)
                        print('method ', method)
                        print('alpha = ', params['alpha'])
                        print('weight decay = ', wd)
                        if network == 'unet':
                            #features=(32, 32, 32, 64, 128, 32)
                            if method == 'standard':
                                net = BasicUNet(spatial_dims=2, out_channels=1, features=(32, 32, 32, 64, 128, 32), norm=("group", {"num_groups": 4}), act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}), dropout=params['dropout']).to(device)
                                ModelParamsInit(net)
                            if method == 'SGLD':
                                net = BasicUNet(spatial_dims=2, out_channels=1, features=(32, 32, 32, 64, 128, 32), norm=None, act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01})).to(device)
                                ModelParamsInit(net)
                        
                        criterion = nn.L1Loss()
                      
                        if opt == 'ADAM':
                            optimizer=torch.optim.Adam(net.parameters(), lr=learn, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
                        
                        [trainloader_1, trainloader_2] = trainloaders(params, data)
                        train_test_net(trainloader_1, trainloader_2, network, N_finish, N, params, params['alpha'], learn, method, optimizer, criterion, net, device, l, wd)
  
if(t==2):
    data = pd.read_pickle("./data_50.pkl", compression='infer')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N_fin = [60, 80, 100, 120] #number of iterations
    params = scan_params()

    for dose in params['dose_list']:
        data_d = data[data['Dose']==dose] #doses
        [trainloader_1, trainloader_2] = trainloaders(params, data_d)
        for N_ in N_fin:
            
            PATH = os.path.join(root, 'Experiments_fin_/') #add here the 'path to' for your results
    
            results_summary(trainloader_2, N_, device, PATH, dose)
  
if(t==3): #evaluate data
    
    params = scan_params()
    
    net_path = os.path.join(root, 'Experiments_fin_/SGLD/unet_15_epochs_0.0005_lr_[0.7, 0.3, 0, 0]grad_loss_lambdaweight_decay1e-09/epoch_10_iter_2996/net.pt')
    PATH = os.path.join(root, 'Experiments_fin_/')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
    k=0
    root_dir = os.path.join(root, 'challange_eval/test/')
    pred_dir = os.path.join(root, 'challange_eval/prediction/')
    meta_data_path = os.path.join(root, 'meta_info.csv')
    list_dir = os.listdir(root_dir)
    for data_dir in list_dir:
        print(data_dir)
        if os.path.isdir(os.path.join(root_dir, data_dir)):
            files = os.listdir(os.path.join(root_dir, data_dir))
            for file in files:
                pred = nib.load(os.path.join(root_dir, data_dir, file)).get_fdata()
                img = []
                s = pred.shape
                print(s)
                
                l=s[2]
                for i in range(0, l):
                    
                    slice_ = pred[:,:,i]
                   
                    inputs = get_slice_ready(device, slice_)
                    net = torch.load(net_path).to(device)
                    net.eval()
                    results = net(inputs)
                    result = results.detach().cpu().squeeze(0).squeeze(0).numpy()
                    
                    result_fin = un_norm(result, slice_)
                    
                    img.append(result_fin)
                    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
                    ax1.imshow(slice_)
                    ax2.imshow(result_fin)
                    plt.close(f)
            
                img_fin = np.swapaxes(np.array(img), 0, 2)
                print(img_fin.shape)
                save_nift(np.array(img_fin), os.path.join(pred_dir, data_dir))


            
        