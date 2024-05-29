import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from datetime import date, timedelta
import numpy as np


class GAN_dataset(Dataset):
    """
    A PyTorch Dataset for multi-modal GAN training from GLORYS data.

    Attributes:
        data_dir (str): The directory containing all input data files.
        latent_dim (int): Dimension of random latent vector inputted to gnerator
        n_lon (int): Longitude size of NN reconstruction patch
        n_lat (int): Latitude size of NN reconstruction patch
        samples_per_day (int): Number of patches to take on each day (randomly distributed)
        date_range (list): List of 2 datetime objects for start and end date of sampling period.
        variables (list): List of strings with the names of the data variables to extract from the dataset e.g. 'zos', 'so','thetao', 'uo', 'vo'
        var_stds (dictionary): Dictionary with std to normalize by for each variable in variables
        model_zarr_name (str): Name of the Zarr store for the model (within data_dir)
        lon_buffers (list): Number of pixels in the longitude direction at each edge to exclude from the sampling. None = no buffer.
        lat_buffers (list): Number of pixels in the latitude direction at each edge to exclude from the sampling. None = no buffer.
        multiprocessing (bool): Indicates if multi-processing will be used in dataloader. Needed to initialize dataset otherwise it's done in worker_init_fn. (default: True)
        
    Methods:
        __len__(): Returns the number of samples in the dataset.
        worker_init_fn(worker_id): Initializes worker processes for multiprocessing.
        __getitem__(idx): Returns a tuple containing input and output data for the given index.

            Input Data (invar): A tensor of shape latent_dim containing Gaussian random noise for generator input.
            Output Data (outvar): A tensor of shape (len(variables), n_lon, n_lat) containing extracted Glorys data.
    """
    
    def __init__(self, data_dir, latent_dim, n_lon, n_lat, samples_per_day, date_range, variables, var_stds, model_zarr_name, lon_buffers = [None, None], lat_buffers = [None, None], multiprocessing = True):
        self.data_dir = data_dir
        self.latent_dim = latent_dim
        self.n_lon = n_lon
        self.n_lat = n_lat
        self.samples_per_day = samples_per_day
        self.lon_buffers = lon_buffers
        self.lat_buffers = lat_buffers
        self.date_range = date_range
        self.variables = variables
        self.var_stds = var_stds
        self.n_channels = len(variables)
        self.model_zarr_name = model_zarr_name
        
        if not multiprocessing:
            self.ds_model = xr.open_zarr(self.data_dir + self.model_zarr_name)
            
            i_lon_min = self.lon_buffers[0]
            i_lon_max = -self.lon_buffers[1]
            i_lat_min = self.lat_buffers[0]
            i_lat_max = -self.lat_buffers[1]
                
            self.ds_model = self.ds_model.isel(longitude = slice(i_lon_min, i_lon_max), latitude = slice(i_lat_min, i_lat_max), depth = 0, drop = True).sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
            
            self.N_lon = self.ds_model.dims['longitude']
            self.N_lat = self.ds_model.dims['latitude']
            self.N_time = self.ds_model.dims['time']
            
            # Generate isel arguments
            self.indexer = []
            np.random.seed(1)
            for t in range(self.N_time):
                lat_indices = np.random.choice(self.N_lat - self.n_lat + 1, self.samples_per_day, replace=False)
                lon_indices = np.random.choice(self.N_lon - self.n_lon + 1, self.samples_per_day, replace=False)
                for i in range(lon_indices.shape[0]):# lat_start in lat_indices:
                    # for lon_start in lon_indices:
                    self.indexer.append({
                        'latitude': slice(lat_indices[i], lat_indices[i] + self.n_lat),
                        'longitude': slice(lon_indices[i], lon_indices[i] + self.n_lon),
                        'time': t
                    })
            
            
        
    def __len__(self):
        return int(self.samples_per_day*(self.date_range[1]-self.date_range[0]).days)
    
    def worker_init_fn(self, worker_id):
        self.ds_model = xr.open_dataset(self.data_dir + self.model_zarr_name, engine='zarr')

        i_lon_min = self.lon_buffers[0]
        if self.lon_buffers[1] is not None:
            i_lon_max = -self.lon_buffers[1]
        else:
            i_lon_max = None
        i_lat_min = self.lat_buffers[0]
        if self.lat_buffers[1] is not None:
            i_lat_max = -self.lat_buffers[1]
        else:
            i_lat_max = None

        self.ds_model = self.ds_model.isel(longitude = slice(i_lon_min, i_lon_max), latitude = slice(i_lat_min, i_lat_max), depth = 0, drop = True).sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)

        self.N_lon = self.ds_model.dims['longitude']
        self.N_lat = self.ds_model.dims['latitude']
        self.N_time = self.ds_model.dims['time']

        # Generate isel arguments
        self.indexer = []
        np.random.seed(1)
        for t in range(self.N_time):
            lat_indices = np.random.choice(self.N_lat - self.n_lat + 1, self.samples_per_day, replace=False)
            lon_indices = np.random.choice(self.N_lon - self.n_lon + 1, self.samples_per_day, replace=False)
            for i in range(lon_indices.shape[0]):# lat_start in lat_indices:
                # for lon_start in lon_indices:
                self.indexer.append({
                    'latitude': slice(lat_indices[i], lat_indices[i] + self.n_lat),
                    'longitude': slice(lon_indices[i], lon_indices[i] + self.n_lon),
                    'time': t
                })
        

    def __getitem__(self, idx):
        
        invar = torch.from_numpy(np.random.randn(self.latent_dim))
        
        data_model = self.ds_model.isel(self.indexer[idx])
        
        outvar = np.zeros((self.n_channels, self.n_lat, self.n_lon))
        for v, var in enumerate(self.variables):
            outvar[v,] = data_model[var]/self.var_stds[var]
        
        outvar = torch.from_numpy(outvar.astype(np.float32))
        outvar = torch.nan_to_num(outvar, nan=0.0)
        
        return invar, outvar