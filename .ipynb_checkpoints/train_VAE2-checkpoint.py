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
import math
import csv

class MultiHeadDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels, num_heads, output_size, hidden_dim=256):
        super(MultiHeadDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.output_size = output_size
        self.hidden_dim = hidden_dim

        # Shared initial layers, directly outputting the right shape for heads
        self.shared_layers = nn.Sequential(
            nn.Linear(latent_dim, num_heads * hidden_dim),
            nn.ReLU()
        )

        # Calculate starting size and upscaling steps for ConvTranspose2d
        start_size = 4  
        upscale_factor = int(math.log2(output_size / start_size))
        if 2**upscale_factor != output_size / start_size:
            raise ValueError(f"Output size {output_size} must be a power of 2 multiple of the minimum size {start_size}")

        # Define individual decoder heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                # Additional ConvTranspose2d layers if needed
                *[nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1, bias=False) for _ in range(upscale_factor - 1)],
                nn.ConvTranspose2d(hidden_dim, output_channels, 4, stride=2, padding=1, bias=False),  # Last layer for output
                # One more layer for the extra factor of 2
                nn.ConvTranspose2d(output_channels, output_channels, 4, stride=2, padding=1, bias=False) 
            ) for _ in range(num_heads)
        ])
    def forward(self, z):
        # Pass through shared layers 
        z = self.shared_layers(z)

        # Reshape for decoder heads 
        z = z.view(-1, self.num_heads, self.hidden_dim, 1, 1)

        # Decode with each head
        outputs = [head(z[:, i, :]) for i, head in enumerate(self.heads)]

        # Concatenate head outputs along the channel dimension
        return torch.cat(outputs, dim=1) 

class MultiHeadEncoder(nn.Module):
    def __init__(self, latent_dim, input_channels, num_heads, input_size, hidden_dim):
        super(MultiHeadEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.num_heads = num_heads
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        # Downsampling factor calculation, mirroring the decoder's upscaling
        start_size = 4
        downscale_factor = int(math.log2(input_size / start_size))

        # Ensure input size is compatible
        if 2**downscale_factor != input_size / start_size:
            raise ValueError(f"Input size {input_size} must be a power of 2 multiple of {start_size}")

        # Individual encoder heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                # Initial Conv2d layer, mirroring the decoder's last
                nn.Conv2d(input_channels, hidden_dim, 4, stride=2, padding=1, bias=False),
                # Extra Conv2d for the additional factor of 2
                nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1, bias=False),
                # Conv2d layers for downsampling
                *[nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1, bias=False) for _ in range(downscale_factor - 1)], 
                nn.BatchNorm2d(hidden_dim),  # BatchNorm for stability
                nn.ReLU(),                 # Activation
                nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1, bias=False), # Final Conv2d to match hidden_dim
            ) for _ in range(num_heads)
        ])

        # Shared final layers, outputting the latent representation
        self.shared_layers = nn.Sequential(
            nn.ReLU(),  # Activation, mirroring the decoder
            nn.Linear(num_heads * hidden_dim, 2*latent_dim) 
        )

    def forward(self, x):
        # Split the input across the channel dimension for multi-head processing
        head_inputs = x.chunk(self.num_heads, dim=1) 

        # Process each head individually
        head_outputs = [head(x) for x, head in zip(head_inputs, self.heads)]  

        # Flatten and concatenate head outputs
        x = torch.cat([h.view(h.size(0), -1) for h in head_outputs], dim=1) 

        # Pass through shared layers for final latent representation
        x = self.shared_layers(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_channels, output_channels, latent_dim, hidden_dim=256, input_size=64, output_size=64, num_heads=3):
        super(VAE, self).__init__()
        self.encoder = MultiHeadEncoder(latent_dim, input_channels, num_heads, input_size, hidden_dim)
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        self.decoder = MultiHeadDecoder(latent_dim, output_channels, num_heads, output_size, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_logvar = self.encoder(x).view(-1, 2, self.latent_dim)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
class LossLoggerCallback:
    def __init__(self, filename):
        self.filename = filename
        self.losses = []
        self.recon_losses = []
        self.kld_losses = []

    def __call__(self, epoch, loss, recon_loss, kld_loss):
        self.losses.append(loss)
        self.recon_losses.append(recon_loss)
        self.kld_losses.append(kld_loss)

        # Save the losses to a CSV file
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Loss', 'ReconLoss', 'KLDLoss'])
            for i in range(len(self.losses)):
                writer.writerow([i+1, self.losses[i], self.recon_losses[i], self.kld_losses[i]])
    
data_dir = './input_data/'
ds = xr.open_dataset(data_dir + 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_70.00W-40.00W_25.00N-45.00N_0.49m_2010-01-01-2020-12-31.nc')
ds_m = xr.open_dataset(data_dir + 'glorys_gulfstream_means.nc')
ds_clim = xr.open_dataset(data_dir + 'glorys_gulfstream_climatology.nc')

var_stds = {'zos':((ds['zos']-ds_m['zos']).std()), 'thetao':((ds['thetao']-ds_clim['thetao']).std()), 'so':((ds['so']-ds_clim['so']).std()), 'uo':((ds['uo']-ds_m['uo']).std()), 'vo':((ds['vo']-ds_m['vo']).std())}

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

dataset = GAN_dataset(data_dir = './input_data/', latent_dim = 256, n_lon = 128, n_lat = 128, samples_per_day = 10, date_range = [date(2015,1,1),date(2020,12,31)], variables = ['zos', 'thetao', 'so', 'uo', 'vo'], var_stds = var_stds, model_zarr_name = 'glorys_gulfstream_anomaly_zarr', lon_buffers = [3, None], lat_buffers = [None, 2], multiprocessing = True)
batch_size = 32
n_cpus = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn = dataset.worker_init_fn,persistent_workers=True)

vae = VAE(input_channels = 1, output_channels = 1, latent_dim = 256, hidden_dim = 128, input_size=128, output_size=128, num_heads=5).to(device)

optimizer = optim.Adam(vae.parameters(), lr=1e-3)
mse_loss = nn.MSELoss(reduction="mean")  # MSE for VAE loss
weight_dir = './checkpoints/'
log_dir = './logs/'
experiment_name = 'glorys_vae_20150101_20201231_10perday_latentdim256_beta0.1'
beta_kld = 0.1

epochs = 1000
losses = []
recon_losses = []
kld_losses = []

loss_logger = LossLoggerCallback(log_dir + experiment_name + "_losses.csv")

for epoch in range(epochs):
    vae.train()  # Ensure model is in train mode
    train_loss = 0
    recon_loss_tracker = 0
    kld_loss_tracker = 0

    for batch_idx, data in enumerate(dataloader):
        data = data[1].to(device)

        optimizer.zero_grad()
        recon_x, mu, logvar = vae(data)

        # Reconstruction Loss (MSE)
        recon_loss = mse_loss(recon_x, data)

        # KL Divergence Loss (regularization)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total Loss
        loss = recon_loss + beta_kld * kld_loss

        loss.backward()
        nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        train_loss += loss.item()  # Accumulate loss
        recon_loss_tracker += recon_loss.item()
        kld_loss_tracker += kld_loss.item()
    train_loss /= len(dataloader)
    recon_loss_tracker /= len(dataloader)
    kld_loss_tracker /= len(dataloader)

    losses.append(train_loss)#/(len(dataset)*5*128*128))
    print(f'Epoch: {epoch}/{epochs}, Loss: {train_loss/(len(dataloader)*batch_size)}')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,#/(len(dataset)*5*128*128),
        'recon_loss': recon_loss_tracker,#/(len(dataset)*5*128*128),
        'kld_loss': kld_loss_tracker,#/(len(dataset)*5*128*128)
    }

    torch.save(checkpoint, weight_dir+experiment_name+f'_weights_epoch{epoch}')
    loss_logger(epoch, train_loss, recon_loss_tracker, kld_loss_tracker)
    
    
