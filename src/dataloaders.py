import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from src.interp_utils import *
from datetime import date, timedelta
import os
import numpy as np


class NeurOST_dataset(Dataset):
    """
    A PyTorch Dataset for loading and preparing SSH and SST data for NeurOST inference.

    Attributes:
        data_dir (str): The directory containing all input data files.
        mid_date (datetime.date): The middle date of the time period to extract data, i.e. the desired reconstruction date.
        N_t (int): The number of time steps (days) to include in the dataset.
        mean_ssh (float): Mean SSH value for standardization.
        std_ssh (float): Standard deviation of SSH values for standardization.
        mean_sst (float): Mean SST value for standardization.
        std_sst (float): Standard deviation of SST values for standardization.
        coord_grids (np.ndarray): Coordinates for the output subdomain grids.
        n (int, optional): Number of bins in the output grid (default: 128).
        L_x (float, optional): Longitudinal range of the output grid in meters (default: 960e3).
        L_y (float, optional): Latitudinal range of the output grid in meters (default: 960e3).
        filtered_sla (bool, optional): Whether to use filtered sea level anomaly (SLA) data (default: False).
        withheld_sats (list, optional): List of satellite identifiers to withhold from the data (default: None).
        sst_zarr_name (str, optional): Name of the Zarr file containing SST data (default: 'foo_bar').
        multiprocessing (bool, optional): Whether to use multiprocessing for data loading (default: True).
        
    Methods:
        __len__(): Returns the number of samples in the dataset.
        worker_init_fn(worker_id): Initializes worker processes for multiprocessing.
        __getitem__(idx): Returns a tuple containing input and output data for the given index.

            Input Data (invar): A tensor of shape (N_t, 2, n, n) containing gridded SST and SSH data.
            Output Data (outvar): A tensor containing SSH data.
    """
    
    def __init__(self, data_dir, mid_date, N_t, mean_ssh, std_ssh, mean_sst, std_sst, coord_grids, n = 128, L_x = 960e3, L_y = 960e3, filtered_sla = False, withheld_sats = None, sst_zarr_name = 'foo_bar', multiprocessing = True):
        self.data_dir = data_dir
        self.date = date
        self.N_t = N_t
        self.mean_ssh = mean_ssh
        self.std_ssh = std_ssh
        self.mean_sst = mean_sst
        self.std_sst = std_sst
        self.coord_grids = coord_grids
        self.n = n
        self.L_x = L_x
        self.L_y = L_y
        self.start_date = mid_date - timedelta(days=N_t/2)
        self.dates = [self.start_date + timedelta(days = t) for t in range(N_t)]
        ssh_files = GetListOfFiles(data_dir+'cmems_sla/')
        self.ssh_files = [f for f in ssh_files if date(int(f[-20:-16]),int(f[-16:-14]),int(f[-14:-12])) in self.dates]
        self.max_outvar_length = 400
        self.filtered_sla = filtered_sla
        self.withheld_sats = withheld_sats
        self.sst_zarr_name = sst_zarr_name
        if not multiprocessing:
            self.worker_ds_sst = xr.open_dataset(self.data_dir + self.sst_zarr_name, engine='zarr')
            self.worker_ds_ssh =load_multisat_ssh(self.ssh_files)
        
    def __len__(self):
        return self.coord_grids.shape[0]
    
    def worker_init_fn(self, worker_id):
        self.worker_ds_sst = xr.open_dataset(self.data_dir + self.sst_zarr_name, engine='zarr')
        self.worker_ds_ssh =load_multisat_ssh(self.ssh_files)

    def __getitem__(self, idx):
        lon0 = 0.25*(self.coord_grids[idx,int(self.n/2)-1,int(self.n/2)-1,0]+self.coord_grids[idx,int(self.n/2)-1,int(self.n/2),0]+self.coord_grids[idx,int(self.n/2),int(self.n/2)-1,0]+self.coord_grids[idx,int(self.n/2),int(self.n/2),0])
        lat0 = 0.25*(self.coord_grids[idx,int(self.n/2)-1,int(self.n/2)-1,1]+self.coord_grids[idx,int(self.n/2)-1,int(self.n/2),1]+self.coord_grids[idx,int(self.n/2),int(self.n/2)-1,1]+self.coord_grids[idx,int(self.n/2),int(self.n/2),1])
        
        sst = grid_sst_hr(self.worker_ds_sst,self.N_t,self.n, self.L_x, self.L_y, lon0, lat0, self.coord_grids[idx,])

        sst[sst!=0] = (sst[sst!=0]-self.mean_sst)/self.std_sst
        
        if self.withheld_sats is not None:
            tracks_in, tracks_out = extract_tracked(self.worker_ds_ssh, self.start_date, self.start_date+timedelta(days=self.N_t+1), self.L_x, self.L_y, lon0, lat0, transformer_ll2xyz, self.coord_grids[idx,], self.filtered_sla, withhold_sats = self.withheld_sats)
        else:
            tracks_in = extract_tracked(self.worker_ds_ssh, self.start_date, self.start_date+timedelta(days=self.N_t+1), self.L_x, self.L_y, lon0, lat0, transformer_ll2xyz, self.coord_grids[idx,], self.filtered_sla, withhold_sats = self.withheld_sats)
        ssh_in = grid_ssh(tracks_in, self.n, self.N_t, self.L_x, self.L_y, self.start_date)
        

        ssh_in[ssh_in!=0] = (ssh_in[ssh_in!=0]-self.mean_ssh)/self.std_ssh
        invar = torch.from_numpy(np.stack((sst, ssh_in), axis = 1).astype(np.float32))
        if self.withheld_sats is not None:
            ssh_out = reformat_output_tracks(tracks_out, self.max_outvar_length, self.N_t, self.n, self.L_x, self.L_y, self.start_date, self.mean_ssh, self.std_ssh, self.filtered_sla)
            outvar = torch.from_numpy(ssh_out)
        else:
            outvar = torch.from_numpy(np.zeros(1).astype(np.float32))
        
        return invar, outvar