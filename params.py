#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:52:30 2021

@author: maya.fi@bm.technion.ac.il
"""

def scan_params():
    params=dict()

    params['drf'] = 50
    params['alpha'] = 1e-4
    params['dose_list'] = ['50']
    params['chop'] = 0
    params['real_list'] = ['A']
    params['multi_slice_n']= 1
    params['new_h'] = 128
    params['new_w']= 128  
    params['train_val_test'] = [0.2,0.05] #split of pt. between train_test
    params['batch_size'] = 1
    params['ker_size'] = 3
  
    params['num_chan'] = 1
    params['num_kernels'] = 3
    params['num_of_epochs'] = 35
    
    params['lr'] = [5e-4]
    
    params['momentum'] = 0.9
    params['dropout'] = 0.2
   
    params['net'] = ['unet']
    params['weight_decay'] = [1e-9] #best
    params['gain'] = 1
    params['t'] = 5
    params['lambda'] = [[1-0.3, 0.3]]
    params['optimizer']=['ADAM']
    params['N_finish'] = 15
    params['method']=['SGLD']
    return params
