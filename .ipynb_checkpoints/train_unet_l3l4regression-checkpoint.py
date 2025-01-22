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
import json
from src.unet import *

device = torch.device('cuda:0')

with open('/nobackup/samart18/GenDA/input_data/diffusion_training_rescale_factors.json', 'r') as f:
    rescale_factors = json.load(f)


data_dir = '/nobackup/samart18/GenDA/input_data/'
variables_in = ['zos', 'thetao', 'uas', 'vas']
variables_oi = ['ssh_oi', 'sst_oi', 'sss_oi']
variables_out = ['zos','thetao','so','u_ageo_eddy', 'v_ageo_eddy']

batch_size = 64
n_cpus = 4

buffers = 12

dataset = L3L4_Regression_Training_Dataset(data_dir = '/nobackup/samart18/GenDA/input_data/', 
                                           n_lon = 128, 
                                           n_lat = 128, 
                                           date_range = [date(2010,1,1),date(2016,12,31)], 
                                           variables_in = variables_in, 
                                           variables_oi = variables_oi, 
                                           variables_out = variables_out, 
                                           var_stds = rescale_factors, 
                                           lon_buffers = [buffers, buffers], 
                                           lat_buffers = [buffers, buffers + 6], 
                                           multiprocessing = False, 
                                           add_obs_noise = True, 
                                           noise_attenuation_factor = 0.2
                                          )

dataset_sampler = InfiniteSampler(
    dataset=dataset, rank=0, num_replicas=1, seed=0
)

dataset_iter = iter(DataLoader(dataset, sampler = dataset_sampler, batch_size=batch_size))

# dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus, worker_init_fn = dataset.worker_init_fn,persistent_workers=True)


dataset_val = L3L4_Regression_Training_Dataset(data_dir = '/nobackup/samart18/GenDA/input_data/', 
                                           n_lon = 128, 
                                           n_lat = 128, 
                                           date_range = [date(2018,1,1),date(2020,12,31)], 
                                           variables_in = variables_in, 
                                           variables_oi = variables_oi, 
                                           variables_out = variables_out, 
                                           var_stds = rescale_factors, 
                                           lon_buffers = [buffers, buffers], 
                                           lat_buffers = [buffers, buffers + 6], 
                                           multiprocessing = False, 
                                           add_obs_noise = True, 
                                           noise_attenuation_factor = 0.2
                                          )


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

loss_logger = LossLoggerCallback("unet_l3l4_regression_losses_noislands_final_wnoise.csv")

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
    torch.save(model.state_dict(), f'unet_checkpoints/UNet_l3l4regression_wSWOT_oi_ssh25_sst16_sss16_noislands_final_wnoise_epoch{epoch}.pt')

losses = np.stack((np.array(train_loss_tracker), np.array(val_loss_tracker)), axis = 0)
np.save('unet_l3l4regression_losses_noislands_final_wnoise.npy', losses)
