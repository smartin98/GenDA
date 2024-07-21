import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from datetime import date, timedelta
import numpy as np


class VAE_dataset(Dataset):
    """
    A PyTorch Dataset for multi-modal VAE training from GLORYS data.

    Attributes:
        data_dir (str): The directory containing all input data files.
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

    Returns:
        Input/Output Data (outvar): A tensor of shape (len(variables), n_lon, n_lat) containing extracted Glorys data - input = output for VAE.
    """
    
    def __init__(self, data_dir, n_lon, n_lat, samples_per_day, date_range, variables, var_stds, model_zarr_name, lon_buffers = [None, None], lat_buffers = [None, None], multiprocessing = True):
        self.data_dir = data_dir
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
            self.ds_model = xr.open_dataset(self.data_dir + 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_70.00W-40.00W_25.00N-45.00N_0.49m_2010-01-01-2020-12-31.nc')
            ds_m = xr.open_dataset(self.data_dir + 'glorys_gulfstream_means.nc')
            monthly_climatology = xr.open_dataset(self.data_dir + 'glorys_gulfstream_climatology.nc')
            self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']
            self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']
            for var in ['zos','uo','vo']:
                self.ds_model[var] = self.ds_model[var] - ds_m[var]

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

            self.indexer = []
            np.random.seed(1)
            for t in range(self.N_time):
                lat_indices = np.random.choice(self.N_lat - self.n_lat + 1, self.samples_per_day, replace=False)
                lon_indices = np.random.choice(self.N_lon - self.n_lon + 1, self.samples_per_day, replace=False)
                for i in range(lon_indices.shape[0]):#
                    self.indexer.append({
                        'latitude': slice(lat_indices[i], lat_indices[i] + self.n_lat),
                        'longitude': slice(lon_indices[i], lon_indices[i] + self.n_lon),
                        'time': t
                    })
            
            
        
    def __len__(self):
        return int(self.samples_per_day*(self.date_range[1]-self.date_range[0]).days)
    
    def worker_init_fn(self, worker_id):
        self.ds_model = xr.open_dataset(self.data_dir + 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_70.00W-40.00W_25.00N-45.00N_0.49m_2010-01-01-2020-12-31.nc')
        ds_m = xr.open_dataset(self.data_dir + 'glorys_gulfstream_means.nc')
        monthly_climatology = xr.open_dataset(self.data_dir + 'glorys_gulfstream_climatology.nc')
        self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']
        self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']
        for var in ['zos','uo','vo']:
            self.ds_model[var] = self.ds_model[var] - ds_m[var]

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

        self.indexer = []
        np.random.seed(1)
        for t in range(self.N_time):
            lat_indices = np.random.choice(self.N_lat - self.n_lat + 1, self.samples_per_day, replace=False)
            lon_indices = np.random.choice(self.N_lon - self.n_lon + 1, self.samples_per_day, replace=False)
            for i in range(lon_indices.shape[0]):
                self.indexer.append({
                    'latitude': slice(lat_indices[i], lat_indices[i] + self.n_lat),
                    'longitude': slice(lon_indices[i], lon_indices[i] + self.n_lon),
                    'time': t
                })

    def update_indexes(self):
        self.indexer = []
        for t in range(self.N_time):
            lat_indices = np.random.choice(self.N_lat - self.n_lat + 1, self.samples_per_day, replace=False)
            lon_indices = np.random.choice(self.N_lon - self.n_lon + 1, self.samples_per_day, replace=False)
            for i in range(lon_indices.shape[0]):
                self.indexer.append({
                    'latitude': slice(lat_indices[i], lat_indices[i] + self.n_lat),
                    'longitude': slice(lon_indices[i], lon_indices[i] + self.n_lon),
                    'time': t
                })
        

    def __getitem__(self, idx):
                
        data_model = self.ds_model.isel(self.indexer[idx])
        
        outvar = np.zeros((self.n_channels, self.n_lat, self.n_lon))
        for v, var in enumerate(self.variables):
            outvar[v,] = data_model[var]/self.var_stds[var]
        
        outvar = torch.from_numpy(outvar.astype(np.float32))
        outvar = torch.nan_to_num(outvar, nan=0.0)
        
        return outvar

