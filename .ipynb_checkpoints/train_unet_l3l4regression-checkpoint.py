import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import xarray as xr
import numpy as np
import sys
sys.path.append('src')
from src.dataloaders import *
sys.path.append('/nobackup/samart18/modulus')
from modulus.utils.generative import InfiniteSampler
from tqdm import tqdm
import torch.optim as optim

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))

class UNet_large(nn.Module):
    def __init__(self, in_channels=7, out_channels=5, dropout_rate=0.2):
        super(UNet_large, self).__init__()
        
        # Encoder
        self.down1 = DownsampleBlock(in_channels, 32, dropout_rate)
        self.down2 = DownsampleBlock(32, 64, dropout_rate)
        self.down3 = DownsampleBlock(64, 128, dropout_rate)
        self.down4 = DownsampleBlock(128, 256, dropout_rate)

        self.res1 = ResBlock(256, 256, dropout_rate)
        self.res2 = ResBlock(256, 256, dropout_rate)
        
        # Decoder
        self.up1 = UpsampleBlock(256, 128, dropout_rate)
        self.up2 = UpsampleBlock(256, 64, dropout_rate)
        self.up3 = UpsampleBlock(128,32, dropout_rate)
        self.up4 = UpsampleBlock(64, 32, dropout_rate)
        
        # Final convolution
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        # Self-attention
        # sa = self.self_attention(d4)
        
        # Residual blocks
        r = self.res1(d4)
        r = self.res2(r)
        
        # Decoder
        u1 = self.up1(r)
        u1 = torch.cat([u1, d3], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u4 = self.up4(u3)
        
        # Final convolution
        output = self.final_conv(u4)
        
        return output

device = torch.device('cuda:0')



data_dir = '/nobackup/samart18/GenDA/input_data/'
ds = xr.open_dataset(data_dir + 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_70.00W-40.00W_25.00N-45.00N_0.49m_2010-01-01-2020-12-31_wERA5_winds_and_geostrophy_15m_ekman_regression.nc')
ds_m = xr.open_dataset(data_dir + 'glorys_gulfstream_means_wERA5_winds_and_geostrophy_15m_ekman_regression.nc')
ds_clim = xr.open_dataset(data_dir + 'glorys_gulfstream_climatology.nc')

var_stds = {'zos':float((ds['zos']-ds_m['zos']).std()), 
            'thetao':float((ds['thetao'].groupby('time.month')-ds_clim['thetao']).std()), 
            'so':float((ds['so'].groupby('time.month')-ds_clim['so']).std()), 
            'u_ageo_eddy':float((ds['u_ageo_eddy']-ds_m['u_ageo_eddy']).std()), 
            'v_ageo_eddy':float((ds['v_ageo_eddy']-ds_m['v_ageo_eddy']).std()), 
            'uas':float((ds['uas']-ds_m['uas']).std()), 
            'vas':float((ds['vas']-ds_m['vas']).std())}

variables_in = ['zos', 'thetao', 'uas', 'vas']
variables_oi = ['ssh_oi', 'sst_oi', 'sss_oi']
variables_out = ['zos','thetao','so','u_ageo_eddy', 'v_ageo_eddy']



batch_size = 64
n_cpus = 4

dataset = L3obs_plus_L4OI_Regression(data_dir = '/nobackup/samart18/GenDA/input_data/', n_lon = 128, n_lat = 128, date_range = [date(2010,1,1),date(2016,12,31)], variables_in = variables_in, variables_oi = variables_oi, variables_out = variables_out, var_stds = var_stds, model_zarr_name = 'glorys_gulfstream_anomaly_zarr', lon_buffers = [3, None], lat_buffers = [None, 2], multiprocessing = False)

dataset_sampler = InfiniteSampler(
    dataset=dataset, rank=0, num_replicas=1, seed=0
)

dataset_iter = iter(DataLoader(dataset, sampler = dataset_sampler, batch_size=batch_size))

# dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn = dataset.worker_init_fn,persistent_workers=True)


dataset_val = L3obs_plus_L4OI_Regression(data_dir = '/nobackup/samart18/GenDA/input_data/', n_lon = 128, n_lat = 128, date_range = [date(2018,1,1),date(2020,12,31)], variables_in = variables_in, variables_oi = variables_oi, variables_out = variables_out, var_stds = var_stds, model_zarr_name = 'glorys_gulfstream_anomaly_zarr', lon_buffers = [3, None], lat_buffers = [None, 2], multiprocessing = False)


dataset_val_sampler = InfiniteSampler(
    dataset=dataset_val, rank=0, num_replicas=1, seed=0
)

valid_dataset_iter = iter(DataLoader(dataset_val, sampler = dataset_val_sampler, batch_size=batch_size))

model = UNet_large(dropout_rate = 0).to(device)


criterion = nn.MSELoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 100


class LossLoggerCallback:
    def __init__(self, filename):
        self.filename = filename
        self.train_losses = []
        self.val_losses = []

    def __call__(self, epoch, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Save the losses to a CSV file
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
            for i in range(len(self.train_losses)):
                writer.writerow([i+1, self.train_losses[i], self.val_losses[i]])


train_loss_tracker = []
val_loss_tracker = []

loss_logger = LossLoggerCallback("unet_l3l4_regression_losses.csv")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    # Training loop
    for batch in tqdm(range(100), desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        inputs, targets = next(dataset_iter)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    avg_train_loss = total_loss / batch_count

    print(avg_train_loss)
    train_loss_tracker.append(avg_train_loss)
    
    # Validation loop
    model.eval()
    total_val_loss = 0
    val_batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(range(100), desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            inputs, targets = next(valid_dataset_iter)
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            
            total_val_loss += val_loss.item()
            val_batch_count += 1
    
    avg_val_loss = total_val_loss / val_batch_count
    print(avg_val_loss)
    loss_logger(epoch, avg_train_loss, avg_val_loss)
    val_loss_tracker.append(avg_val_loss)
    torch.save(model.state_dict(), f'unet_checkpoints/UNet_l3l4regression_wSWOT_wnoise_oi_ssh25_sst16_sss16_epoch{epoch}.pt')

losses = np.stack((np.array(train_loss_tracker), np.array(val_loss_tracker)), axis = 0)
np.save('unet_l3l4regression_losses.npy')
