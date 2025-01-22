import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
import datetime
import sys
sys.path.append('/nobackup/samart18/modulus')
sys.path.append('src')

from concurrent.futures import ThreadPoolExecutor

import cftime
from einops import rearrange
import hydra
import netCDF4 as nc
import nvtx
from omegaconf import OmegaConf, DictConfig
import torch
from torch.distributed import gather
import torch._dynamo
import tqdm

from src.dataloaders import *

from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import (
    ablation_sampler,
    parse_int_list,
    StackedRandomGenerator,
    InfiniteSampler,
)
from modulus import Module

time_format = "%Y-%m-%dT%H:%M:%S"

import torch.nn as nn
from torch import Size, Tensor
import torch.nn.functional as F
from typing import *
from tqdm import tqdm
import math

from src.sda import *
import json
from glob import glob

def multichannel_gaussian_blur(img, sigmas_rc):
    """
    Applies Gaussian blur to an image with channel and dimension-specific sigmas.
    
    Args:
    img: Input image tensor of shape (B, C, H, W).
    sigmas_rc: List of tuples (sigma_row, sigma_col) for each channel.
    
    Returns:
    Blurred image tensor of the same shape as the input.
    """
    device = img.device
    
    B, C, H, W = img.shape
    out = torch.zeros_like(img, device = device)
    
    for c in range(C):
        sigma_r, sigma_c = sigmas_rc[c]
        
        # Create 1D Gaussian kernels
        kernel_size_r = int(sigma_r * 3) * 2 + 1
        kernel_size_c = int(sigma_c * 3) * 2 + 1
        kernel_r = torch.exp(-torch.pow(torch.arange(kernel_size_r) - (kernel_size_r - 1) / 2, 2) / (2 * sigma_r**2)).to(device)
        kernel_r = kernel_r / kernel_r.sum()
        kernel_c = torch.exp(-torch.pow(torch.arange(kernel_size_c) - (kernel_size_c - 1) / 2, 2) / (2 * sigma_c**2)).to(device)
        kernel_c = kernel_c / kernel_c.sum()
        
        # Create 2D Gaussian kernel
        kernel_2d = torch.outer(kernel_r, kernel_c)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        # Apply Gaussian blur
        out[:,c] = F.conv2d(img[:,c].unsqueeze(1), kernel_2d, padding='same')[:,0,]
    
    return out

data_dir = '/nobackup/samart18/GenDA/input_data/'

with open('/nobackup/samart18/GenDA/input_data/diffusion_training_rescale_factors.json', 'r') as f:
    rescale_factors = json.load(f)

ds_masks = xr.open_dataset(data_dir + 'obs_masks_ssh-sst-u-v.nc')
ds_masks['ssh_total'] = (ds_masks['ssh_nadir'].astype('bool')) | (ds_masks['ssh_karin'].astype('bool'))
ds_oi = xr.open_dataset(data_dir + 'oi_l4_ssh-sst-sss_full_domain_updated_with_errors_sigmas_ssh25_sst16_sss16_norescaling_nodemean_noislands.nc')
ds_m = xr.open_dataset(data_dir + 'glorys_means_pre_processed_fixed_noislands.nc')
ds_clim = xr.open_dataset(data_dir + 'glorys_climatology_pre_processed_fixed_noislands.nc')

ds_oi['ssh_oi'] = (ds_oi['ssh_oi'] - ds_m['zos'])/rescale_factors['zos']
ds_oi['sst_oi'] = ((ds_oi['sst_oi'].groupby('time.month') - ds_clim['thetao'])/rescale_factors['thetao']).drop('month')
ds_oi['sss_oi'] = ((ds_oi['sss_oi'].groupby('time.month') - ds_clim['so'])/rescale_factors['so']).drop('month')
ds_oi['ssh_oi_standard_error'] = ds_oi['ssh_oi_standard_error']/rescale_factors['zos']
ds_oi['sst_oi_standard_error'] = ds_oi['sst_oi_standard_error']/rescale_factors['thetao']
ds_oi['sss_oi_standard_error'] = ds_oi['sss_oi_standard_error']/rescale_factors['so']

sigma_L_ssh = 25#26.5 # sigma of SSH Gaussian smoothing kernel in km (see Table S4 in https://doi.org/10.31223/X5W676)
sigma_T_ssh = 7/4 # sigma of SSH Gaussian smoothing kernel in days (assuming smoothing for time-scales ~1 week)
sigma_L_sst = 16 # sigma of SST Gaussian smoothing kernel in km (as used in https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004047)
sigma_T_sst = 1.23 # sigma of SST Gaussian smoothing kernel in days (as used in https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004047)
sigma_L_sss = 16 #sigma of SSS Gaussian smoothing kernel in km (assume same as SST)
sigma_T_sss = 1.23 # sigma of SSS Gaussian smoothing kernel in days (assume same as SST)
NN_res = 1/12
# convert these to lat-lon pixels:
deg_lon_in_km = 6378*2*np.pi*np.cos(np.deg2rad(38))/360
deg_lat_in_km = 6378*2*np.pi/360
sigma_lon_ssh = (1 / NN_res) * sigma_L_ssh / deg_lon_in_km
sigma_lon_sst = (1 / NN_res) * sigma_L_sst / deg_lon_in_km
sigma_lon_sss = (1 / NN_res) * sigma_L_sss / deg_lon_in_km

sigma_lat_ssh = (1 / NN_res) * sigma_L_ssh / deg_lat_in_km
sigma_lat_sst = (1 / NN_res) * sigma_L_sst / deg_lat_in_km
sigma_lat_sss = (1 / NN_res) * sigma_L_sss / deg_lat_in_km

lon_min = -65
lon_max = -55
lat_min = 33
lat_max = 43
time_min = '2017-01-01'
time_max = '2018-01-01'

NN_res = 1/12
NN_input_size = 128

buffer_lon = int((NN_input_size - abs(lon_max - lon_min) / NN_res) / 2)
buffer_lat = int((NN_input_size - abs(lat_max - lat_min) / NN_res) / 2)

lon_min_NN, lon_max_NN = lon_min - buffer_lon * NN_res, lon_max + buffer_lon * NN_res
lat_min_NN, lat_max_NN = lat_min - buffer_lat * NN_res, lat_max + buffer_lat * NN_res

ds_oi = ds_oi.sel(longitude = slice(lon_min_NN, lon_max_NN - NN_res), latitude = slice(lat_min_NN, lat_max_NN - NN_res))
ds_oi = ds_oi.sel(time = slice(time_min, time_max))

ds_oi['ssh_oi'][:,:,:8] = np.nan
ds_oi['ssh_oi'][:,:,-8:] = np.nan
ds_oi['ssh_oi'][:,:8,:] = np.nan
ds_oi['ssh_oi'][:,-8:,:] = np.nan

ds_oi['sst_oi'][:,:,:8] = np.nan
ds_oi['sst_oi'][:,:,-8:] = np.nan
ds_oi['sst_oi'][:,:8,:] = np.nan
ds_oi['sst_oi'][:,-8:,:] = np.nan

ds_oi['sss_oi'][:,:,:8] = np.nan
ds_oi['sss_oi'][:,:,-8:] = np.nan
ds_oi['sss_oi'][:,:8,:] = np.nan
ds_oi['sss_oi'][:,-8:,:] = np.nan


print('loading dataset')

dataset = GenDA_OSSE_Inference_Dataset(data_dir = '/nobackup/samart18/GenDA/input_data/', 
                       lon_min = lon_min_NN, 
                       lon_max = lon_max_NN - NN_res,
                       lat_min = lat_min_NN, 
                       lat_max = lat_max_NN - NN_res,
                       input_dim = (128, 128),
                       date_range = [date(2017,1,1),date(2017,12,31)], 
                       variables = ['zos', 'thetao', 'so', 'u_ageo_eddy', 'v_ageo_eddy', 'uas', 'vas'], 
                       var_stds = rescale_factors, 
                       multiprocessing = False
                        )

print('dataset loaded')

batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

n_cpus = 1
DistributedManager.initialize()
dist = DistributedManager()
device = dist.device

files = glob('/nobackup/samart18/GenDA/outputs/batch64_songunet_vanilla_noislands_rescale/output_diffusion/ema*')


res_ckpt_filename = sorted(files)[-1]
reg_ckpt_filename = "./checkpoints/regression.mdlus"
seeds = "0-63"
sample_seeds = parse_int_list(seeds)
image_outdir = "image_outdir"

class_idx = None  # TODO: is this needed
num_steps = 8
sample_res = "full"
sampling_method = "stochastic"
seed_batch_size = 1
force_fp16 = True
use_torch_compile = False
regression_only = False
diffusion_only = True

# Parse deterministic sampler options
sigma_min = None
sigma_max = None
rho = 7
solver = "heun"
discretization = "edm"
schedule = "linear"
scaling = None
S_churn = 0
S_min = 0
S_max = float("inf")
S_noise = 1

# Parse data options
times_range = None
patch_size = 448
patch_shape_x = None
patch_shape_y = None
overlap_pix = 0
boundary_pix = 0
hr_mean_conditioning = False

img_shape_x = 128
img_shape_y = 128

if sampling_method == "stochastic":
    sampler_kwargs = {
        "img_shape": img_shape_x,
        "patch_shape": patch_shape_x,
        "overlap_pix": overlap_pix,
        "boundary_pix": boundary_pix,
    }

if (patch_shape_x is None) or (patch_shape_x > img_shape_x):
    patch_shape_x = img_shape_x
if (patch_shape_y is None) or (patch_shape_y > img_shape_y):
    patch_shape_y = img_shape_y
if patch_shape_x != img_shape_x or patch_shape_y != img_shape_y:
    if patch_shape_x != patch_shape_y:
        raise NotImplementedError("Rectangular patch not supported yet")
    if patch_shape_x % 32 != 0 or patch_shape_y % 32 != 0:
        raise ValueError("Patch shape needs to be a multiple of 32")
    if sampling_method == "deterministic":
        raise NotImplementedError(
            "Patch-based deterministic sampler not supported yet. Please use stochastic sampler instead. "
        )
    print("Patch-based generation enabled")
else:
    print("Patch-based generation disabled")

# Sanity check for the type of requested inference
if regression_only and diffusion_only:
    raise ValueError(
        "Both regression_only and diffusion_only cannot be set to True."
    )
if regression_only:
    net_res = None
if diffusion_only:
    net_reg = None

# Load diffusion network, move to device, change precision
if not regression_only:
    print(f'Loading residual network from "{res_ckpt_filename}"...')
    net_res = Module.from_checkpoint(res_ckpt_filename)
    net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
    if force_fp16:
        net_res.use_fp16 = True

# load regression network, move to device, change precision
if not diffusion_only:
    print(f'Loading network from "{reg_ckpt_filename}"...')
    net_reg = Module.from_checkpoint(reg_ckpt_filename)
    net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
    if force_fp16:
        net_reg.use_fp16 = True

# Reset since we are using a different mode.
if use_torch_compile:
    torch._dynamo.reset()
    compile_mode = "reduce-overhead"
    # Only compile residual network
    # Overhead of compiling regression network outweights any benefits
    if net_res:
        net_res = torch.compile(net_res, mode=compile_mode)


ds_noise = xr.open_dataset('input_data/OSE_L3_products.nc')

#ose errors:
ds_noise = xr.open_dataset('input_data/OSE_L3_products.nc')

attenuation_factor = 0.2

ssh_obs_noise_std = float(ds_noise['ssh_error'].mean()) * attenuation_factor
sst_obs_noise_std = float(ds_noise['sst_error'].mean()) * attenuation_factor
wind_noise_level = 1e-2

std_ssh_oi = float(ds_oi['ssh_oi'].std())
std_sst_oi = float(ds_oi['sst_oi'].std())
std_sss_oi = float(ds_oi['sss_oi'].std())

pred_dir = f'predictions/batch64_songunet_vanilla_ema_noislands_osse_wnoise_ideal_drifters_attenuation{attenuation_factor}_winds_noiselevel1e-2/'

if ~(os.path.isdir(pred_dir)):
    os.mkdir(pred_dir)
    
os.mkdir('osse_preds')

n_members = 24
start_date = datetime.date(2017,1,1)

eps = eps_edm(net_res, shape = ())

for t in range(365):
    _, x_star = dataset.__getitem__(t)
    x_star = x_star.unsqueeze(0)

    ssh_mean = torch.from_numpy(dataset.ds_m['zos'].values)#.to(device)
    sst_mean = torch.from_numpy(dataset.monthly_climatology['thetao'].isel(month = int(dataset.ds_model.isel(time=t)['time.month'].values) - 1,drop = True).values)#.to(device)
    sss_mean = torch.from_numpy(dataset.monthly_climatology['so'].isel(month = int(dataset.ds_model.isel(time=t)['time.month'].values) - 1,drop = True).values)#.to(device)

    oi_ground_truth = torch.from_numpy(np.expand_dims(np.stack((ds_oi['ssh_oi'].isel(time = t),
                           ds_oi['sst_oi'].isel(time = t),
                           ds_oi['sss_oi'].isel(time = t)), axis = 0), axis = 0))

    oi_ground_truth_err = torch.from_numpy(np.expand_dims(np.stack((ds_oi['ssh_oi_standard_error'].isel(time = t),
                               ds_oi['sst_oi_standard_error'].isel(time = t),
                               ds_oi['sss_oi_standard_error'].isel(time = t)), axis = 0), axis = 0))
    
    oi_mask = np.stack((~np.isnan(ds_oi['ssh_oi'].isel(time = t)), ~np.isnan(ds_oi['sst_oi'].isel(time = t)), ~np.isnan(ds_oi['sss_oi'].isel(time = t))), axis = 0).astype('bool')
    
    total_mask = np.stack((ds_masks['ssh_total'].isel(time = t), 
                           ds_masks['sst_mask'].isel(time = t), 
                           np.zeros((128,128)),
                           ds_masks['drifter_mask'].isel(time = t),
                           ds_masks['drifter_mask'].isel(time = t),
                           np.ones((128,128)),
                           np.ones((128,128)),
                          ), axis = 0)

    n_obs = int(np.sum(total_mask[:5,]))

    noise = np.zeros((7,128,128))
    noise_levels = np.zeros((7,128,128))
    noise[0,] = np.random.randn(128,128) * ssh_obs_noise_std
    noise[1,] = np.random.randn(128,128) * sst_obs_noise_std
    noise_levels[0,:,:] = ssh_obs_noise_std
    noise_levels[1,:,:] = sst_obs_noise_std
    noise_levels[5,] = wind_noise_level
    noise_levels[6,] = wind_noise_level

    def A(x):
        inst_obs = x[:, total_mask.astype('bool')]

        ssh = x[:,0:1,].clone() * rescale_factors['zos'] + ssh_mean.to(x.device)
        sst = x[:,1:2,].clone() * rescale_factors['thetao'] + sst_mean.to(x.device)
        sss = x[:,2:3,].clone() * rescale_factors['so'] + sss_mean.to(x.device)
        
        # smooth background SSH, SST, SSS from OI:
        smoothed_obs = multichannel_gaussian_blur(torch.concat((ssh,sst,sss), axis = 1), 
                                                 sigmas_rc = [(sigma_lat_ssh, sigma_lon_ssh), 
                                                             (sigma_lat_sst, sigma_lon_sst), 
                                                             (sigma_lat_sss, sigma_lon_sss)]
                                                 )

        smoothed_obs[:,0,] = (smoothed_obs[:,0,] - ssh_mean.to(x.device))/rescale_factors['zos']
        smoothed_obs[:,1,] = (smoothed_obs[:,1,] - sst_mean.to(x.device))/rescale_factors['thetao']
        smoothed_obs[:,2,] = (smoothed_obs[:,2,] - sss_mean.to(x.device))/rescale_factors['so']
        
        smoothed_obs = smoothed_obs[:,oi_mask]
        
        return torch.concat((inst_obs, smoothed_obs),axis=1)

    inst_obs = x_star[0, total_mask.astype('bool')] + noise[total_mask.astype('bool')]
    inst_obs = inst_obs.reshape(1,inst_obs.shape[0]).repeat(n_members,1)
    inst_obs_noise_level = torch.from_numpy(noise_levels[total_mask.astype('bool')])
    inst_obs_noise_level = inst_obs_noise_level.reshape(1,inst_obs_noise_level.shape[0]).repeat(n_members,1)
    
    n_i_obs = torch.numel(inst_obs)

    oi_ground_truth = oi_ground_truth.repeat(n_members,1,1,1)
    oi_ground_truth = oi_ground_truth[:,oi_mask]
    n_s = torch.numel(oi_ground_truth)
    oi_ground_truth = torch.nan_to_num(oi_ground_truth, 0)

    oi_ground_truth_err = oi_ground_truth_err.repeat(n_members,1,1,1)
    oi_ground_truth_err = oi_ground_truth_err[:,oi_mask]
    oi_ground_truth_err = torch.nan_to_num(oi_ground_truth_err, 0)
 
    # create the ground truth by applying the observation operator to the simulation and using the simulated L4 products
    y = A(torch.zeros((n_members,7,128,128)))
    y[:,:-oi_ground_truth.shape[1]] = inst_obs
    y[:,-oi_ground_truth.shape[1]:] = oi_ground_truth

    std = A(torch.zeros((n_members,7,128,128)))
    std[:,:-oi_ground_truth.shape[1]] = inst_obs_noise_level
    std[:,-oi_ground_truth.shape[1]:] = oi_ground_truth_err

    sde = VPSDE(
        GaussianScore(
            y,
            A=A,
            std=std,
            sde=VPSDE(eps, shape=()),
            gamma = 1e-1,
        ),
        shape=x_star.shape[1:],
        outer = True,
    ).cuda()
    # make predictions:
    x = sde.sample((n_members,), steps=256, corrections=0, tau=0.3).cpu().numpy()

    # save predictions:
    np.save(pred_dir + f'pred{t}.npy', x)


    # plotting:
    oi_ground_truth = np.expand_dims(np.stack((ds_oi['ssh_oi'].isel(time = t),
                               ds_oi['sst_oi'].isel(time = t),
                               ds_oi['sss_oi'].isel(time = t)), axis = 0), axis = 0)

    fig, axs = plt.subplots(8,7,figsize = (20,20), constrained_layout = True)
    
    variables = ['SSH', 'SST', 'SSS', '$u_{ageo}$', '$v_{ageo}$', '$u_{atmos}$', '$v_{atmos}$']
    rows = ['Observed', 'OI', 'Ground Truth','Prediction Mean', 'Prediction Std', 'RMSE', 'Ensemble Mean RMSE', 'Prediction (1st Member)']
    cmaps = [cmocean.cm.curl, cmocean.cm.thermal, cmocean.cm.haline, cmocean.cm.balance, cmocean.cm.balance, cmocean.cm.delta, cmocean.cm.delta]
    
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            if i == 0:
                if j in [0,1,5,6]:
                        im = axs[i,j].pcolormesh(dataset.ds_model['longitude'], dataset.ds_model['latitude'], (x_star[0,j,]+noise[j,])*total_mask[j,], cmap = cmaps[j], vmin = -3, vmax = 3)
                        axs[i,j].set_title(variables[j] + ' ' + rows[i])
                else:
                    fig.delaxes(axs[i,j])
            elif i == 1:
                if j in [0,1,2]:
                    axs[i,j].pcolormesh(dataset.ds_model['longitude'], dataset.ds_model['latitude'], oi_ground_truth[0,j,], cmap = cmaps[j], vmin = -3, vmax = 3)
                    axs[i,j].set_title(variables[j] + ' ' + rows[i])
                else:
                    fig.delaxes(axs[i,j])
            elif i == 2:
                axs[i,j].pcolormesh(dataset.ds_model['longitude'], dataset.ds_model['latitude'], x_star[0,j,], cmap = cmaps[j], vmin = -3, vmax = 3)
                axs[i,j].set_title(variables[j] + ' ' + rows[i])
            elif i == 3:
                axs[i,j].pcolormesh(dataset.ds_model['longitude'], dataset.ds_model['latitude'], np.mean(x[:,j,],axis=0), cmap = cmaps[j], vmin = -3, vmax = 3)
                axs[i,j].set_title(variables[j] + ' ' + rows[i])
            elif i == 4:
                axs[i,j].pcolormesh(dataset.ds_model['longitude'], dataset.ds_model['latitude'], np.std(x[:,j,], axis=0), cmap = cmaps[j], vmin = -3, vmax = 3)
                axs[i,j].set_title(variables[j] + ' ' + rows[i])
            elif i == 5:
                axs[i,j].pcolormesh(dataset.ds_model['longitude'], dataset.ds_model['latitude'], np.sqrt(np.mean((x[:,j,] - x_star[0,j,].detach().cpu().numpy())**2, axis=0)), cmap = cmaps[j], vmin = -3, vmax = 3)
                axs[i,j].set_title(variables[j] + ' ' + rows[i])
            elif i == 6:
                axs[i,j].pcolormesh(dataset.ds_model['longitude'], dataset.ds_model['latitude'], np.sqrt((np.mean(x[:,j,], axis=0) - x_star[0,j,].detach().cpu().numpy())**2), cmap = cmaps[j], vmin = -3, vmax = 3)
                axs[i,j].set_title(variables[j] + ' ' + rows[i])
            else:
                axs[i,j].pcolormesh(dataset.ds_model['longitude'], dataset.ds_model['latitude'], x[i,j,], cmap = cmaps[j], vmin = -3, vmax = 3)
                axs[i,j].set_title(variables[j] + ' ' + rows[i])
    fig.suptitle(f'{start_date + datetime.timedelta(days = t)}')
    cbar = fig.colorbar(im,ax = axs, ticks=[-3, -2, -1, 0, 1, 2, 3], location='bottom', shrink = 0.25)
    cbar.set_label('Standard Deviations', fontsize = 16)
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(f'osse_preds/plot{t}.png')
    plt.close('all')

