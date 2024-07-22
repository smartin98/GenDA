import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
import numpy as np
from datetime import date, timedelta
import sys
sys.path.append('src')
from src.dataloaders import *
from src.vae_model import *
from src.torch_utils import LossLoggerCallback, spectral_loss
import math
import csv
    
    
data_dir = './input_data/'
ds = xr.open_dataset(data_dir + 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_70.00W-40.00W_25.00N-45.00N_0.49m_2010-01-01-2020-12-31.nc')
ds_m = xr.open_dataset(data_dir + 'glorys_gulfstream_means.nc')
ds_clim = xr.open_dataset(data_dir + 'glorys_gulfstream_climatology.nc')

var_stds = {'zos':((ds['zos']-ds_m['zos']).std()), 'thetao':((ds['thetao']-ds_clim['thetao']).std()), 'so':((ds['so']-ds_clim['so']).std()), 'uo':((ds['uo']-ds_m['uo']).std()), 'vo':((ds['vo']-ds_m['vo']).std())}

ngpu = 1
device = torch.device("cuda:0") if (torch.cuda.is_available() and ngpu > 0) else torch.device("cpu")

dataset = VAE_dataset(data_dir = './input_data/', latent_dim = 1024, n_lon = 128, n_lat = 128, samples_per_day = 10, date_range = [date(2015,1,1),date(2020,12,31)], variables = ['zos', 'thetao', 'so', 'uo', 'vo'], var_stds = var_stds, model_zarr_name = 'glorys_gulfstream_anomaly_zarr', lon_buffers = [3, None], lat_buffers = [None, 2], multiprocessing = True)
batch_size = 32
n_cpus = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn = dataset.worker_init_fn,persistent_workers=True)

vae = VAE(input_channels = 1, output_channels = 1, latent_dim = 1024, hidden_dim = 256, input_size=128, output_size=128, num_heads=5).to(device)

optimizer = optim.Adam(vae.parameters(), lr=1e-3)
mse_loss = nn.MSELoss(reduction="mean")  # MSE for VAE loss
weight_dir = './checkpoints/'
log_dir = './logs/'
experiment_name = 'glorys_vae_20150101_20201231_10perday_latentdim1024_hid256_beta0.1_nobnorm_specloss_betaspec0.1_randomized_tanh'
beta_kld = 0.1
beta_spec = 0.1

epochs = 1000
losses = []
recon_losses = []
kld_losses = []

loss_logger = LossLoggerCallback(log_dir + experiment_name + "_losses.csv")

start_epoch = 0

for epoch in range(start_epoch,epochs):
    vae.train()  # Ensure model is in train mode
    train_loss = 0
    recon_loss_tracker = 0
    spec_loss_tracker = 0
    kld_loss_tracker = 0

    for batch_idx, data in enumerate(dataloader):
        data = data[1].to(device)

        optimizer.zero_grad()
        recon_x, mu, logvar = vae(data)

        # Reconstruction Loss (MSE)
        recon_loss = mse_loss(recon_x, data)

        # Reconstruction spectral loss

        spec_loss = spectral_loss(recon_x, data)

        # KL Divergence Loss (regularization)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total Loss
        loss = recon_loss + beta_spec*spec_loss + beta_kld * kld_loss

        loss.backward()
        nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        train_loss += loss.item()  # Accumulate loss
        recon_loss_tracker += recon_loss.item()
        spec_loss_tracker += spec_loss.item()
        kld_loss_tracker += kld_loss.item()
    train_loss /= len(dataloader)
    recon_loss_tracker /= len(dataloader)
    kld_loss_tracker /= len(dataloader)
    spec_loss_tracker/= len(dataloader)

    dataloader.dataset.update_indexes()

    losses.append(train_loss)#/(len(dataset)*5*128*128))
    print(f'Epoch: {epoch}/{epochs}, Loss: {train_loss}')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,#/(len(dataset)*5*128*128),
        'recon_loss': recon_loss_tracker,#/(len(dataset)*5*128*128),
        'kld_loss': kld_loss_tracker,#/(len(dataset)*5*128*128)
    }
    if epoch%10==0:
        torch.save(checkpoint, weight_dir+experiment_name+f'_weights_epoch{epoch}')
    loss_logger(epoch, train_loss, recon_loss_tracker, spec_loss_tracker, kld_loss_tracker)
    
    
