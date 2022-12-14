#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:49:10 2021

@author: maya.fi@bm.technion.ac.il
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

import pandas as pd

import os
import torch.nn

import matplotlib
matplotlib.use('agg')
import warnings
warnings.filterwarnings("ignore")
plt.ioff()   # interactive mode

import albumentations as A
from pydicom import dcmread
import pickle
import torch.nn as nn

import kornia
import torch
import monai
import unet_2

import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau

from glob import glob
import nibabel as nib
torch.set_printoptions(precision=10)

def SUV_convert(result_fin, meta_data_path, file):
    info = pd.read_csv(meta_data_path)
    print('file', file)
    weight = info[info.PID==file].weight
    dose = info[info.PID==file].dose
    
    ratio = int(dose)/int(weight)
    #print(ratio)
    res = result_fin/ratio
    return res


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            #print("yes")
            m.train()  


def results_summary(trainloader_2, N_fin, device, PATH, dose):
    #

    df = pd.DataFrame(columns=['method', 'LDPT', 'NDPT', 'idx', 'mean', 'std', 'ssim_0', 'ssim_net'])
    k=0
   
    for i, data in enumerate(trainloader_2, 0):
         #if i in index_list:
        inputs = data['LDPT'].to(device)
        outputs = data['NDPT']
        
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
 
        for direct in os.listdir(PATH):
            
            if direct == 'SGLD':
                #print(direct)
                j=0
                df_SGLD = pd.DataFrame(columns=['method', 'epoch', 'idx', 'LDPT', 'NDPT', 'NET'])
                for epoch_test in os.listdir(os.path.join(PATH, direct)):
                    for epoch_num in os.listdir(os.path.join(PATH, direct, epoch_test)):
                        l = len(os.listdir(os.path.join(PATH, direct, epoch_test)))
                        path = os.path.join(os.path.join(PATH, direct, epoch_test, epoch_num), 'net.pt')
                        if os.path.isfile(path) and j<N_fin:
                            print('yes')
                            net = torch.load(path).to(device)
                            net.eval()
                            NET = net(inputs).detach().cpu()
                            temp = {'method': direct, 'epoch': epoch_num, 'idx': i, 'LDPT':inputs.detach().cpu().squeeze().numpy(), 'NDPT':outputs.squeeze().numpy(), 'NET':NET.squeeze().numpy()}
                            df_SGLD.loc[j] = temp
                            j=j+1
                            
                    df_return  = dict_mean(df_SGLD, i)
                    #print(len(df_SGLD))
                    temp = {'method': direct, 'LDPT':inputs.detach().cpu().squeeze().numpy(), 'NDPT':outputs.squeeze().numpy(), 'idx':i, 'mean':df_return['mean'], 'std':df_return['std'], 'ssim_0':df_return['ssim_0'], 'ssim_net':df_return['ssim_net']}
                    df.loc[k] = temp
                    k=k+1
                    out_dirs = os.path.join(PATH, direct, epoch_test, 'STD_maps_recon', str(dose), str(N_fin)+'iters')
                    #save_images(df_return, out_dirs)
                    #print(df_return['std'].iloc[0].values.shape)
                    std_ = df_return['std'].iloc[0]
                    ax1.imshow(std_)
                    ax1.set_title('SGLD\n min=' + str(format(std_.min(), '.2E')) + 'max=' + str(format(std_.max(), '.2E')))
   
            if direct == 'standard':
                #print(direct)    
                for epoch_test in os.listdir(os.path.join(PATH, direct)):
                        epoch_num = os.listdir(os.path.join(PATH, direct, epoch_test))[-1]
                        df_standard = pd.DataFrame(columns=['method', 'epoch', 'idx', 'LDPT', 'NDPT', 'NET'])
                        path = os.path.join(os.path.join(PATH, direct, epoch_test, epoch_num), 'net.pt')
                        if os.path.isfile(path):
                            for j in range(0, N_fin):
                                net = torch.load(path).to(device)         
                                net.eval()
                                enable_dropout(net)
                                NET = net(inputs).detach().cpu()
                                temp = {'method': direct, 'epoch': j, 'idx': i, 'LDPT':inputs.detach().cpu().squeeze().numpy(), 'NDPT':outputs.squeeze().numpy(), 'NET':NET.squeeze().numpy()}
                                df_standard.loc[j] = temp
    
                        df_return = dict_mean(df_standard, i)
                        #print(len(df_standard))
                        temp = {'method': direct, 'LDPT':inputs.detach().cpu(), 'NDPT':outputs, 'idx':i, 'mean':df_return['mean'], 'std':df_return['std'], 'ssim_0':df_return['ssim_0'], 'ssim_net':df_return['ssim_net']}
                        df.loc[k] = temp
                        k=k+1
                        out_dirs = os.path.join(PATH, direct, epoch_test, epoch_num, 'STD_maps_recon', str(dose), str(N_fin)+'iters')
                        #save_images(df_return, out_dirs)
                        std_ = df_return['std'].iloc[0]
                        ax2.imshow(std_)
                        ax2.set_title('Dropout\n min=' + str(format(std_.min(), '.2E')) + 'max=' + str(format(std_.max(), '.2E')))
                        if_no_dir_mkdir(os.path.join(PATH, "STD maps", str(dose), str(N_fin) + 'iters'))
                        f.savefig(os.path.join(PATH, "STD maps", str(dose), str(N_fin) + 'iters', "img" + str(i) + ".jpg"))
                        plt.close(f)
                
    df.to_pickle(os.path.join(PATH, str(dose) + 'DRF' + str(N_fin) + 'iters' + 'results.pkl'), compression='infer', protocol=5, storage_options=None)
    B = df.groupby("method")["ssim_net"].apply(lambda x: np.median([*x], axis=0))
    print("number of iters ", str(N_fin))
    print("dose ",str(dose))
    print(B)

def save_images(df_return, out_dirs):
  
     if not os.path.exists(out_dirs):
            os.makedirs(out_dirs)
     for index, row in df_return.iterrows():
        #print(row['std'])
       
   
        f, (ax1, ax2, ax4) = plt.subplots(1, 3, sharey=True)
    
        ax1.imshow(row['LDPT'])
        ax1.set_title('LDPT' + ' SSIM = {0:.5f}'.format(row['ssim_0']))
        
        ax2.imshow(row['NDPT'])
        ax2.set_title('NDPT')
        
        #ax3.imshow(row['std'])
        #ax3.set_title('std')
        
        ax4.imshow(row['mean'])
        ax4.set_title('mean' + ' SSIM = {0:.5f}'.format(row['ssim_net']))

        f.savefig(os.path.join(out_dirs, "img" + str(row['idx']) + ".jpg"))
        plt.close(f)

def get_slice_ready(device, slice_):
    X = norm_data(slice_).astype(np.float32)
    inputs = torch.tensor(X).unsqueeze(0).unsqueeze(0).to(device)
    return(inputs)
def if_no_dir_mkdir(path):
    isExist = os.path.exists(path)

    if not isExist:
  
  # Create a new directory because it does not exist 
      os.makedirs(path)

def pack_dcm(dcm_dir, out_dir, sub):
    slices = os.listdir(dcm_dir)
    img = []
    for slice_ in slices:
        X = dcmread(os.path.join(dcm_dir, slice_))
        img.append(X.pixel_array)
    if len(sub)==26:
        img_fin = np.swapaxes(np.array(img), 0, 2)
    else:
        img_fin=img
    
    print(np.array(img_fin).shape)
    save_nift(np.array(img_fin), os.path.join(out_dir, sub))

def save_nift(numpy_array, data_path):
    if_no_dir_mkdir(data_path)
    
    ni_img = nib.Nifti1Image(numpy_array, affine=np.eye(4))
    nib.save(ni_img, os.path.join(data_path, 'recon.nii.gz'))
    
def dict_mean(df, i):
    #print(len(df))
    df_return = pd.DataFrame(columns=['LDPT', 'NDPT', 'idx', 'mean', 'std', 'ssim_0', 'ssim_net'])
        
    df_temp = df
    mean = df_temp['NET'].to_numpy().mean(axis=0) 
    std = df_temp['NET'].to_numpy().std(axis=0)
    print(std.shape)
    #print(df_temp.iloc[0].LDPT.shape)
    ssim_0 = ssim(stand_data(torch.tensor(df_temp.iloc[0].LDPT)).numpy(), stand_data(torch.tensor(df_temp.iloc[0].NDPT)).numpy())
    ssim_net = ssim(stand_data(torch.tensor(mean)).numpy(), stand_data(torch.tensor(df_temp.iloc[0].NDPT)).numpy())
    data = {'LDPT': df_temp.iloc[0].LDPT, 'NDPT': df_temp.iloc[0].NDPT, 'idx':i, 'mean': mean, 'std':std, 'ssim_0':ssim_0, 'ssim_net':ssim_net}
    df_return.loc[0] = data
   
        #print(ssim_by_iter)
    return df_return

def dict_mean_s(df, i):
    
    
    df_return = pd.DataFrame(columns=['LDPT', 'NDPT', 'idx', 'mean', 'std'])
    
    df_temp = df
    mean = df_temp['NET'].to_numpy().mean(axis=0) 
    std = df_temp['NET'].to_numpy().std(axis=0)
  
    #ssim_0 = ssim(stand_data(torch.tensor(df_temp.iloc[0].LDPT)).numpy(), stand_data(torch.tensor(df_temp.iloc[0].NDPT)).numpy())
    #ssim_net = ssim(stand_data(torch.tensor(mean)).numpy(), stand_data(torch.tensor(df_temp.iloc[0].NDPT)).numpy())
    data = {'LDPT': df_temp.iloc[0].LDPT, 'NDPT': df_temp.iloc[0].NDPT, 'idx':i, 'mean': mean, 'std':std}
    df_return.loc[0] = data
   
    #print(ssim_by_iter)
    return df_return

def train_test_net(trainloader_1, trainloader_2, network, N_finish, N, params, alpha, learn, method, optimizer, criterion, net, device, l, wd):
   
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    valid_in_ssim = []
    valid_res_ssim = []
             
    valid_loss = []
    train_loss = []
    
    split = np.linspace(0, len(trainloader_1), int(len(trainloader_1)/428)+1)
    split = split[1:]
   
    #print(split)
    for epoch in range(N):
        print('epoch ', epoch+1, ' out of ', N)
        i=0
        iterator=iter(trainloader_1)
        for spl in split:

            running_train_loss = 0.0
            running_g_loss = 0.0
            running_l1_loss = 0.0
            SSIM_LDPT_NDPT_train = []
            SSIM_RESU_NDPT_train = []
            SSIM_LDPT_NDPT_valid = []
            SSIM_RESU_NDPT_valid = []
							
            net.train()
            while(i<=int(spl)-1):

                data_train = next(iterator)
                i=i+1
                inputs = data_train['LDPT'].to(device)
                outputs = data_train['NDPT'].to(device)
            
                optimizer.zero_grad()
                results = net(inputs)
                l1_train = torch.tensor(l[0])*criterion(results, outputs)
                grad_train = torch.tensor(l[1])*criterion(gradient_magnitude(results), gradient_magnitude(outputs))
                
                ssim_value = 1 - calc_ssim(results.detach().cpu(), outputs.detach().cpu())
                if ssim_value > 0.05:
                    loss_train = l1_train + grad_train
                    loss_train.backward()
                    optimizer.step()
                    if method == 'SGLD':
                        for parameters in net.parameters():
                            parameters.grad += torch.tensor(learn*alpha).to(device, non_blocking=True)*torch.randn(parameters.grad.shape).to(device, non_blocking=True)						
                    running_train_loss = running_train_loss + loss_train.item()
                    running_g_loss = running_g_loss + grad_train.item()
                    running_l1_loss = running_l1_loss + l1_train.item()
                    SSIM_LDPT_NDPT_train.append(calc_ssim(inputs.detach().cpu(), outputs.detach().cpu()))
                    SSIM_RESU_NDPT_train.append(calc_ssim(results.detach().cpu(), outputs.detach().cpu()))
       
            running_valid_loss = 0.0
    		
            with torch.no_grad():
                net.eval()
                for i_t, data_val in enumerate(trainloader_2, 0):
                    inputs = data_val['LDPT'].to(device)   
                    outputs = data_val['NDPT'].to(device)
                    results = net(inputs)
                    l1_val = torch.tensor(l[0])*criterion(outputs, results)
                    grad_val = torch.tensor(l[1])*criterion(gradient_magnitude(outputs), gradient_magnitude(results)) 
                    
                    loss_val = l1_val + grad_val
                        
                    								
                    running_valid_loss = running_valid_loss + loss_val.item()
                    SSIM_LDPT_NDPT_valid.append(calc_ssim(inputs.detach().cpu(), outputs.detach().cpu()))
                    SSIM_RESU_NDPT_valid.append(calc_ssim(results.detach().cpu(), outputs.detach().cpu()))
			
                scheduler.step(running_valid_loss)	
                print("learning rate = ", scheduler._last_lr)	           
                print("ssim valid LDPT/NDPT", np.mean(SSIM_LDPT_NDPT_valid))
			
                valid_in_ssim.append(np.mean(SSIM_LDPT_NDPT_valid))
			
                print("ssim valid results/NDPT", np.mean(SSIM_RESU_NDPT_valid))
                valid_res_ssim.append(np.mean(SSIM_RESU_NDPT_valid))
                print('[%d, %5d] training loss: %.5f' %
                						  (epoch + 1, i, running_train_loss))
                print('[%d, %5d] training grad loss: %.5f' %
                						  (epoch + 1, i, running_g_loss))
                print('[%d, %5d] training l1 loss: %.5f' %
                						  (epoch + 1, i, running_l1_loss))
                train_loss.append(running_train_loss)
                print('[%d, %5d] validation loss: %.5f' %
                						  (epoch + 1, i, running_valid_loss))
             
            
                valid_loss.append(running_valid_loss)
               
                if N - epoch <= N_finish:
                    PATH_last_models = os.path.join('Experiments_fin_AE', method, network + '_' + str(params['num_of_epochs']) + "_epochs_" + str(learn) + "_lr_" + str(l) + "grad_loss_lambda" + 'weight_decay' + str(wd), 'epoch_'+str(epoch+1)+'_iter_'+str(i))
                    if not os.path.exists(PATH_last_models):
                        os.makedirs(PATH_last_models)
                    save_run(PATH_last_models, net, train_loss, valid_loss, valid_in_ssim, valid_res_ssim)
                    load_model(epoch+1, PATH_last_models, trainloader_2)   
			
               
    print('Finished Training')



def plot_im(PATH, inputs, outputs, results, i):                                
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    ax1.imshow(inputs[0,0,:,:].detach().cpu())
    ax1.set_title('LDPT')
    
    ax2.imshow(outputs[0,0,:,:].detach().cpu())
    ax2.set_title('NDPT')
    
    ax3.imshow(results[0,0,:,:].detach().cpu())
    ax3.set_title('NET')
 
    f.savefig(os.path.join(PATH, "img" + str(i) + ".jpg"))
    plt.close(f)

def dataframe_paths(PATH):
    l=[]
    df = pd.DataFrame(columns=['sub_ID', 'slice', 'Dose', 'LDPT', 'HDPT', 'scanner'])
    i=0                           
    for path, subdirs, files in os.walk(PATH):
        
        for name in files:
            LD = os.path.join(path, name)
            p = LD.split('/')
            #print(p[7])
            if p[7] == 'LD':
                FD = LD.replace('/LD/', '/FD/')
                data = {'sub_ID': p[6], 'slice': p[8], 'Dose': p[5], 'LDPT':LD, 'HDPT':FD, 'scanner': p[6][0]}
                df.loc[i] = data
                i=i+1
                #print(data)
    return df

def split_df(df, params):
    length = len(df)
    weight_list = params['train_val_test'] 
    lengths = [np.ceil(i*length) for i in weight_list]
    #h_length = [np.ceil(l/2) for l in lengths]
    
    df_U = df
    df_U_idx = df.index.to_list()
    #sublists_U = split(h_length, df_U, df_U_idx)
    sublists_U = split(lengths, df_U, df_U_idx)
    #df_S = df[df['scanner'] == 'S']
    #df_S_idx = df[df['scanner'] == 'S'].index.to_list()
    #sublists_S = split(h_length, df_S, df_S_idx)
    
     
    #return sublists_U[0]+sublists_S[0], sublists_U[1]+sublists_S[1]
    return sublists_U[0], sublists_U[1]

def split(lengths, data):
    sublists = []
    i = 0
    prev_index = 0
    for l in lengths:
        l=int(l)
        if(i==0):
            sublists.append(list(range(0, l-1)))
            
            i=i+1
            prev_index = l-1
        else:
            #curr_index = l + prev_index
            j=prev_index+1
            while(data.iloc[prev_index].sub_ID==data.iloc[j].sub_ID):
                j=j+1
                #print(data.iloc[prev_index].sub_ID)
            sublists.append(list(range(j, j+l-1)))
   
    return sublists                 

          
    
def train_val_test_por(params, data):
    length = len(data)
    #my_list = list(range(1, length))
    weight_list = params['train_val_test'] 
    lengths = [np.ceil(i*length) for i in weight_list]
    
    sublists = split(lengths, data)
    #print()
    return sublists[0], sublists[1]            


def get_mat(name):
    X = dcmread(name)
    X = X.pixel_array
    return torch.tensor(X.astype(float))            
 
def get_mat_compare(name):
    X = dcmread(name)
    X = X.pixel_array
    #print('yes')
    return X.astype(float)     

def transforma():
    transform = A.Compose([A.Flip(p=0.5)], additional_targets={'image0': 'image', 'image1': 'image'})
    return transform

def export_pixel_array(in_file_LD, out_file_LD, in_file_FD, out_file_FD, dataset_name):
    LDPT = dcmread(in_file_LD).pixel_array.astype(float)
    NDPT = dcmread(in_file_FD).pixel_array.astype(float)
    transforms = transforma()
    transformed = transforms(image=LDPT, image0=NDPT)
    LDPT = transformed["image"]
    NDPT = transformed["image0"]
    #print(norm_L.size())
    LDPT = norm_data(LDPT[4:356, 4:356]).astype(np.float32)
    NDPT = norm_data(NDPT[4:356, 4:356]).astype(np.float32)
    
    h5 = h5py.File(out_file_LD)
    h5.create_dataset(dataset_name, data=LDPT)
    h5.close()
    
    h5 = h5py.File(out_file_FD)
    h5.create_dataset(dataset_name, data=NDPT)
    h5.close()
    

def calc_valid_loss(criterion, l, data, device, trainloader_2):
    running_valid_loss = 0.0
    with torch.no_grad():
     
        for i, data in enumerate(trainloader_2, 0):
            
            inputs = data['LDPT'].to(device)
            outputs = data['NDPT'].to(device)
            loss_ = torch.tensor(l[0])*criterion(outputs, inputs)
            loss_grad = torch.tensor(l[1])*criterion(gradient_magnitude(outputs), gradient_magnitude(inputs))
           
           
            loss_v = loss_ + loss_grad    
            #print(loss_v.item())
            running_valid_loss = running_valid_loss + loss_v.item()
        return running_valid_loss

def gradient_magnitude(x):
    x_grad = kornia.filters.spatial_gradient(x, mode='diff', order=1)
    x_grad = torch.squeeze(x_grad)
    x_grad_mag = torch.sqrt(torch.tensor(1e-12)+torch.square(x_grad[0]) + torch.square(x_grad[1]))
    #x_grad_mag = torch.square(x_grad[0]) + torch.square(x_grad[1])
    return x_grad_mag

def laplacian_filter(x, kernel_size):
    #print(x.size())
    x_lap = kornia.filters.laplacian(x, kernel_size)
    #print(x_lap.size())
    return x_lap
    
def ModelParamsInitHelper(m, flag):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and flag == "LeakyRELU":
        #print("init weight LeakyRELU")
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif isinstance(m, nn.Conv2d) and flag == "Tanh":
        #print("update weight conv Tanh")
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
    #elif isinstance(m , (nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
        #print("init weight norm")
        #nn.init.constant_(m.weight, 1.0)
        #nn.init.constant_(m.bias, 0)
   

def ModelParamsInit(model):
    assert isinstance(model, nn.Module)
    for name, m in model.named_modules():
       for name, mm in m.named_modules():
            for name, mmm in mm.named_modules():
                for name, mmmm in mmm.named_modules():
                    if isinstance(mmmm, (monai.networks.blocks.Convolution)):
                        
                        for mmmmm in mmmm:
                            if isinstance(mmmmm, nn.Conv2d):
                                #print("init weight LeakyRELU")
                                torch.nn.init.kaiming_uniform_(mmmmm.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                            else:
                                for mmmmmm in mmmmm:
                                    ModelParamsInitHelper(mmmmmm, "LeakyRELU")
                    elif isinstance(mmmm, (unet_2.fin_conv)):
                        for mmmmm in mmmm:
                            if isinstance(mmmmm, nn.Conv2d):
                                #print("yes")
                                torch.nn.init.xavier_normal_(mmmmm.weight, gain=1.0)
                            else:
                                for mmmmmm in mmmmm:
                                   ModelParamsInitHelper(mmmmmm, "Tanh")   
def ModelParamsInit_unetr(model):
    assert isinstance(model, nn.Module)
    for name, m in model.named_modules():
        #print(m)
       
        for name, mm in m.named_modules():
            for name, mmm in mm.named_modules():
                #print(mmm)
                if isinstance(mmm, (nn.Conv2d, nn.ConvTranspose2d)):
                    ModelParamsInitHelper(mmm, "LeakyRELU")
                    
                
                else:
                    for name, mmmm in mmm.named_modules():
                        if isinstance(mmmm, (monai.networks.blocks.Convolution)):
                            for mmmmm in mmmm:
                                if isinstance(mmmmm, (nn.Conv2d, nn.ConvTranspose2d)):
                                    torch.nn.init.kaiming_uniform_(mmmmm.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                                else:
                                    for mmmmmm in mmmmm:
                                        ModelParamsInitHelper(mmmmmm, "LeakyRELU")
              
def save_list(l, name):
    file_name = name 
    open_file = open(file_name, "wb")
    pickle.dump(l, open_file)
    open_file.close()
    
def load_list(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list
    #print(loaded_list)
    
def save_run(PATH, net, train_loss, valid_loss, valid_in_ssim, valid_res_ssim):
    torch.save(net, os.path.join(PATH, "net.pt"))
    save_list(train_loss, os.path.join(PATH, 'train_loss.pkl'))
    save_list(valid_loss, os.path.join(PATH, 'valid_loss.pkl'))
    save_list(valid_in_ssim, os.path.join(PATH, 'valid_in_ssim.pkl'))
    save_list(valid_res_ssim, os.path.join(PATH, 'valid_res_ssim.pkl'))
    
def test_ssim(LD, FD):
    s = ssim(stand_data(get_mat(LD)).numpy(), stand_data(get_mat(FD)).numpy())
    #print(s)
    #print(get_mat(LD).numpy().dtype)
    if s>0.6 and s<1:
        return 1
    
def arrange_data(params, root_dir):
    
    dose_list = params['dose_list']
    chop = params['chop']

    df = pd.DataFrame(columns=['sub_ID', 'slice', 'Dose', 'LDPT', 'HDPT'])
    i=0
    for sub_dir in glob(f"{root_dir}/*/"):
        for scans in glob(f"{sub_dir}/*/"):
            sub_ID = scans[-6:] #pt. ID
            s_sub_path = os.path.join(sub_dir, scans)
            d_scans = os.listdir(s_sub_path)    
            #print(d_scans)
            FD_scan = [d for d in d_scans if (d[-6:] == 'NORMAL' or d[-6:] == 'normal' or d == 'Full_dose' or d == 'FD')]
            #print(FD_scan)
            for FD in FD_scan:
                FD_path = os.path.join(s_sub_path, FD)
                for Dose in dose_list:

                    LD_scan = [d for d in d_scans if Dose in d][0]
                    LD_path = os.path.join(s_sub_path, LD_scan)
                    slices = os.listdir(LD_path)
                    print('LD', LD_path)
                    print('FD', FD_path)
                    #print(slices)
                    for sl in slices:
                        LD = os.path.join(LD_path, sl)
                        FD = os.path.join(FD_path, sl)
                        LD_to = os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "U"+sub_ID, "LD", sl[0:-4])
                        HD_to = os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "U"+sub_ID, "FD", sl[0:-4])
                        PATHS = [os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "U"+sub_ID, "LD"), os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "U"+sub_ID, "FD")]
                        for PATH in PATHS:
                            if not os.path.exists(PATH):
                                os.makedirs(PATH)
                        if(test_ssim(LD, FD)==1):
                            data = {'sub_ID': sub_ID, 'slice': sl, 'Dose': Dose, 'LDPT':LD_to, 'HDPT':HD_to}
                            df.loc[i] = data
                            i=i+1
                            export_pixel_array(LD, LD_to, FD, HD_to, "dataset")
                        else:
                            print('fail')
                        
                        #print(i)
                    #slices = []
                        
    return df
  
def find_sl_FD(FD_path, ssl):
    slices = os.listdir(FD_path)
   
    for sl in slices:
        
        if sl.split('.')[3] == ssl:
            
            return sl


def arrange_data_siemense(params, root_dir):
    
    dose_list = params['dose_list']
    chop = params['chop']

    df = pd.DataFrame(columns=['sub_ID', 'slice', 'Dose', 'LDPT', 'HDPT'])
    i=0
    for direct in root_dir:
        sub = os.listdir(direct)
        sub = [s for s in sub if s[-1] != 'p']       #remove zipped  
        for sub_dir in sub:
            sub_path = os.path.join(direct, sub_dir)
            sub_sub_path = os.listdir(sub_path)
            sub_sub_path = [s for s in sub_sub_path if s[-1] != 'X']
 
            for scans in sub_sub_path:
                sub_ID = scans[-6:] #pt. ID
                print(sub_ID)
                s_sub_path = os.path.join(direct, sub_dir, scans)
                d_scans = os.listdir(s_sub_path)
                FD_scan = [d for d in d_scans if (d[-6:] == 'NORMAL' or d[-6:] == 'normal' or d == 'Full_dose' or d == 'FD')]
                for FD in FD_scan:
                    FD_path = os.path.join(s_sub_path, FD)
                    for Dose in dose_list:

                        LD_scan = [d for d in d_scans if Dose in d][0]
                        LD_path = os.path.join(s_sub_path, LD_scan)
                        slices = os.listdir(LD_path)
                        #print(len(slices))
                        for sl in slices:
                            ssl = sl.split('.')[3]
                            LD = os.path.join(LD_path, sl)
                            sl_FD = find_sl_FD(FD_path, ssl)
                            FD = os.path.join(FD_path, sl_FD)

                            LD_to = os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "S"+sub_ID, "LD", ssl)
                            HD_to = os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "S"+sub_ID, "FD", ssl)
                            PATHS = [os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "S"+sub_ID, "LD"), os.path.join('/tcmldrive/users/Maya/ULDPT1/', str(Dose), "S"+sub_ID, "FD")]
                            for PATH in PATHS:
                                if not os.path.exists(PATH):
                                    os.makedirs(PATH)
                            if(test_ssim(LD, FD)==1):
                                data = {'sub_ID': sub_ID, 'slice': sl, 'Dose': Dose, 'LDPT':LD_to, 'HDPT':HD_to}
                                df.loc[i] = data
                                i=i+1
                               
                                export_pixel_array(LD, LD_to, FD, HD_to, "dataset")
                            else:
                                print('fail')
                            
                            print(i)
                        #slices = []
                        
    return df
                        

def scale_data(data):
    
    if(torch.std(data)==0):
        std=1
    else:
        std=torch.std(data)
    
    if(torch.max(data)!=0):
     
        data = (data-torch.min(data))/(torch.max(data)-torch.min(data))-torch.tensor(0.5)
   
    return data

def scale_for_ssim(data):
    if(torch.max(data)!=0):
        data = data/torch.max(data)
    return data
def stand_data(data):
    
    if(torch.std(data)==0):
        std=1
    else:
        std=torch.std(data)
    data = (data-torch.mean(data))/(std)
 
    return data
def un_norm(data, data_or):
    
    #norm=torch.norm(torch.tensor(data_or), p='fro', dim=None, keepdim=False, out=None, dtype=None)
    #if(norm==0):
        #norm=1
    #m = torch.mean(torch.tensor(data_or))
    sum_or = data_or.sum(dtype=np.float64)
    sum_res = data.sum(dtype=np.float64)
    if sum_res==0:
        sum_res=1
    res = (data*(sum_or/sum_res)).astype(np.float64)
    print('or_sum', sum_or)
    print('res_sum', res.sum())
    #print(res)
    #res = (torch.tensor(data)*norm + m).numpy()
    #mini=data_or.min()
    #maxi=data_or.max()
    #print(mini)
    #print(maxi)
    #res = mini + ((data-data.min())/(data.max()-data.min()))*(maxi-mini)
    return res

def norm_data(data):
    data = torch.tensor(data)
    norm=torch.norm(data, p='fro', dim=None, keepdim=False, out=None, dtype=None)
    if(norm==0):
        norm=1
    m = torch.mean(data)
    data = (data - m)/norm
    
    return data.numpy()

def calc_ssim(LDPT, NDPT):
    ssim_c = []
 
    s=LDPT.shape
    for i in range(s[0]):
        LD=LDPT[i][0]
        ND=NDPT[i][0]
        LDPT1 = stand_data(LD).numpy()
        NDPT1 = stand_data(ND).numpy()
                
        ssim_=ssim(NDPT1, LDPT1)
        ssim_c.append(ssim_)
        
    return sum(ssim_c)/len(ssim_c)
    

def plot_result(PATH, i, results, inputs, outputs, ssim_LD_ND, ssim_RE_ND):
    #scalebar = ScaleBar(0.08, "log L2 norm", length_fraction=0.25)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    
    ax1.imshow(inputs)
    ax1.set_title('LDPT')
    
    ax2.imshow(outputs)
    ax2.set_title('NDPT')
    
    ax3.imshow(results)
    ax3.set_title('NET')
    

    f.savefig(os.path.join(PATH, "img" + str(i) + ".jpg"))
    plt.close(f)
    
def load_model(N, PATH, trainloader_2):
        net = torch.load(os.path.join(PATH, 'net.pt'))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        std = []
        fig, (ax1, ax2) = plt.subplots(2)
        
        train_loss = load_list(os.path.join(PATH, 'train_loss.pkl'))
        valid_loss = load_list(os.path.join(PATH, 'valid_loss.pkl'))
        valid_in_ssim = load_list(os.path.join(PATH, 'valid_in_ssim.pkl'))
        valid_res_ssim = load_list(os.path.join(PATH, 'valid_res_ssim.pkl'))
        #print(train_loss)
        #print(valid_loss)
        ax1.plot(range(0,len(train_loss)), train_loss, label = "training loss")
        ax1.plot(range(0,len(valid_loss)), valid_loss, label = "validation loss")
        ax1.set(ylabel="loss")
        ax1.legend()
        ax1.set_title("training / validation loss over epochs")
        #axs[0].savefig(PATH + "train_valid_loss" + ".jpg")
        
        ax2.plot(range(0,len(valid_in_ssim)), valid_in_ssim, label = "LDPT/NDPT ssim")
        ax2.plot(range(0,len(valid_res_ssim)), valid_res_ssim, label = "result/NDPT ssim")
        ax2.set(ylabel="ssim")
        ax2.legend()
        ax2.set_title("validation ssim over epochs")
        fig.savefig(os.path.join(PATH, "_valid__loss_ssim.jpg"))
        plt.close(fig)
        for i, data in enumerate(trainloader_2, 0):
          
            inputs = data['LDPT'].to(device)
            results = net(inputs)
           
            inputs = inputs[0,0,:,:].detach().cpu()
            #print(inputs.shape)

            outputs = data['NDPT'].to(device)
            outputs = outputs[0,0,:,:].detach().cpu() 
            results = results[0,0,:,:].detach().cpu() 
           

            if(i%10==1):
                #plot_result(np.log(results-outputs)-np.log(outputs), np.log(inputs-outputs)-np.log(outputs), np.log(results-inputs)-np.log(outputs))
                plot_result(PATH, i, results, inputs, outputs, ssim(inputs.numpy(), outputs.numpy()), ssim(results.numpy(), outputs.numpy()))
        
        #print(median(std))
def arrange_data_old(params, root_dir):
    dose_list = params['dose_list']
    chop = params['chop']

    df = pd.DataFrame(columns=['sub_ID', 'slice', 'Dose', 'LDPT', 'HDPT'])
    i=0
    for sub_dir in glob(f"{root_dir}/*/"):
        for scans in glob(f"{sub_dir}/*/"):
            sub_ID = scans[-6:] #pt. ID
            s_sub_path = os.path.join(sub_dir, scans)
            d_scans = os.listdir(s_sub_path)
            FD_scan = [d for d in d_scans if (d[-6:] == 'NORMAL' or d[-6:] == 'normal' or d == 'Full_dose' or d == 'FD')]
            for FD in FD_scan:
                FD_path = os.path.join(s_sub_path, FD)
                for Dose in dose_list:
                    #print('dose')
                    LD_scan = [d for d in d_scans if Dose in d][0]
                    LD_path = os.path.join(s_sub_path, LD_scan)
                    slices = os.listdir(LD_path)
                    for sl in slices:
                        LD = os.path.join(LD_path, sl)
                        FD = os.path.join(FD_path, sl)
                        if(test_ssim(LD, FD)==1):
                            data = {'sub_ID': sub_ID, 'slice': sl, 'Dose': Dose, 'LDPT':LD, 'HDPT':FD}
                            #print(data)
                            df.loc[i] = data
                            i=i+1
                            
                            #print(i)
                            if i>5000*len(dose_list):
                                print('done')
                                return df
                    
    #return df