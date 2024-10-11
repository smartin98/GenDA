import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from datetime import date, timedelta
import numpy as np


class Diffusion_Training_Dataset(Dataset):
    """
    A PyTorch Dataset for multi-modal diffusion training from GLORYS data.

    Attributes:
        data_dir (str): The directory containing all input data files.
        latent_dim (int): Dimension of random latent vector inputted to gnerator
        n_lon (int): Longitude size of NN reconstruction patch
        n_lat (int): Latitude size of NN reconstruction patch
        samples_per_day (int): Number of patches to take on each day (randomly distributed)
        date_range (list): List of 2 datetime objects for start and end date of sampling period.
        variables (list): List of strings with the names of the data variables to extract from the dataset e.g. 'zos', 'so','thetao', 'uo', 'vo'
        var_stds (dictionary): Dictionary with std to normalize by for each variable in variables
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
    
    def __init__(self, data_dir, n_lon, n_lat, samples_per_day, date_range, variables, var_stds, lon_buffers = [None, None], lat_buffers = [None, None], multiprocessing = True):
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
        
        # load the pre-processed GLORYS data, mean fields, and monthly climatology (generated using pre-processing script):
        self.ds_model = xr.open_dataset(self.data_dir + 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_70.00W-40.00W_25.00N-45.00N_0.49m_2010-01-01-2020-12-31_wERA5_winds_and_geostrophy_15m_ekman_regression.nc')
        ds_m = xr.open_dataset(self.data_dir + 'glorys_gulfstream_means_wERA5_winds_and_geostrophy_15m_ekman_regression.nc')
        monthly_climatology = xr.open_dataset(self.data_dir + 'glorys_gulfstream_climatology.nc')
        
        # de-seasonalize sst and sss
        if 'so' in self.variables:
            self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']
        if 'thetao' in self.variables:
            self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']
        # de-mean all other variables
        for var in [v for v in self.variables if v not in ['so','thetao']]: 
            self.ds_model[var] = self.ds_model[var] - ds_m[var]
        
        # cap selection indices based on buffer sizes
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
            
        
        # subset model to training region    
        self.ds_model = self.ds_model.isel(longitude = slice(i_lon_min, i_lon_max), latitude = slice(i_lat_min, i_lat_max), depth = 0, drop = True).sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
        
        self.N_lon = self.ds_model.dims['longitude']
        self.N_lat = self.ds_model.dims['latitude']
        self.N_time = self.ds_model.dims['time']
            
        
    def __len__(self):
        return int(1e12) # some arbitrary large number. The sampling operation just selects a random spatio-temporal crop from the larger dataset so this number should just be > number of training steps.
    
    def worker_init_fn(self, worker_id):
        # initialize dataset on each worker if multi-processing used.
        self.ds_model = xr.open_dataset(self.data_dir + 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_70.00W-40.00W_25.00N-45.00N_0.49m_2010-01-01-2020-12-31_wERA5_winds_and_geostrophy_15m_ekman_regression.nc')
        ds_m = xr.open_dataset(self.data_dir + 'glorys_gulfstream_means_wERA5_winds_and_geostrophy_15m_ekman_regression.nc')
        monthly_climatology = xr.open_dataset(self.data_dir + 'glorys_gulfstream_climatology.nc')
        if 'so' in self.variables:
            self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']
        if 'thetao' in self.variables:
            self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']
        
        for var in [v for v in self.variables if v not in ['so','thetao']]: #['zos','uo','vo']:
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
        

    def __getitem__(self, idx):
        still_checking = True
        while still_checking:
           # generate random spatiotemporal index for subset
            lat_start = np.random.randint(self.N_lat - self.n_lat + 1)
            lon_start = np.random.randint(self.N_lon - self.n_lon + 1)
            indexer = {
                       'latitude': slice(lat_start, lat_start + self.n_lat),
                       'longitude': slice(lon_start, lon_start + self.n_lon), 
                       'time': np.random.randint(self.N_time)
                      }
            
            data_model = self.ds_model.isel(indexer)
            
            outvar = np.zeros((self.n_channels, self.n_lat, self.n_lon))
            for v, var in enumerate(self.variables):
                outvar[v,] = data_model[var]/self.var_stds[var]
            
            outvar = torch.from_numpy(outvar.astype(np.float32))
            outvar = torch.nan_to_num(outvar, nan=0.0)
            if torch.numel(outvar[outvar==0])/torch.numel(outvar)<0.01: # discard samples with greater than 1% land contamination
                still_checking = False
        
        
        return outvar
    
# difference from training dataset: this one takes fixed test region and draws sequentially in time.
class GenDA_OSSE_Inference_Dataset(Dataset):
    """
    A PyTorch Dataset for multi-modal generative data assimilation inference from GLORYS data. This dataset daily samples in order from a fixed sub-domain of the dataset.

    Attributes:
        data_dir (str): The directory containing all input data files.
        lon_min (float): Longitude min bound of reconstruction domain
        lon_max (float): Longitude max bound of reconstruction domain
        lat_min (float): Latitude min bound of reconstruction domain
        lat_max (float): Latitude max bound of reconstruction domain
        
        input_dim (int or tuple of ints): size of NN reconstruction patch: if int (N,N) if tuple: (N_lat, N_lon)
        date_range (list): List of 2 datetime objects for start and end date of sampling period.
        variables (list): List of strings with the names of the data variables to extract from the dataset e.g. 'zos', 'so','thetao', 'uo', 'vo'
        var_stds (dictionary): Dictionary with std to normalize by for each variable in variables
        multiprocessing (bool): Indicates if multi-processing will be used in dataloader. Needed to initialize dataset otherwise it's done in worker_init_fn. (default: True)
        
    Methods:
        __len__(): Returns the number of samples in the dataset.
        worker_init_fn(worker_id): Initializes worker processes for multiprocessing.
        __getitem__(idx): Returns a tuple containing input and output data for the given index.

            Input Data (invar): A tensor of shape latent_dim containing Gaussian random noise for generator input.
            Output Data (outvar): A tensor of shape (len(variables), n_lon, n_lat) containing extracted Glorys data.
    """
    
    def __init__(self, data_dir, lon_min, lon_max, lat_min, lat_max, input_dim, date_range, variables, var_stds, multiprocessing = False):
        self.data_dir = data_dir
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        if type(input_dim) == 'int':
            self.input_dim = (input_dim, input_dim)
        else:
            self.input_dim = input_dim
            
        self.input_dim = input_dim
        self.date_range = date_range
        self.variables = variables
        self.var_stds = var_stds
        self.n_channels = len(variables)
        self.model_file = 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_70.00W-40.00W_25.00N-45.00N_0.49m_2010-01-01-2020-12-31_wERA5_winds_and_geostrophy_15m_ekman_regression.nc'
        self.mean_file = 'glorys_gulfstream_means_wERA5_winds_and_geostrophy_15m_ekman_regression.nc'
        self.clim_file = 'glorys_gulfstream_climatology.nc'

        self.ds_model = xr.open_dataset(self.data_dir + self.model_file)
        ds_m = xr.open_dataset(self.data_dir + self.mean_file)
        monthly_climatology = xr.open_dataset(self.data_dir + self.clim_file)
        if 'so' in self.variables:
            self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']
        if 'thetao' in self.variables:
            self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']
        
        for var in [v for v in self.variables if v not in ['so','thetao']]:   #['zos','uo','vo']:
            self.ds_model[var] = self.ds_model[var] - ds_m[var]
            
        self.ds_model = self.ds_model.isel(depth=0,drop=True).sel(longitude = slice(self.lon_min, self.lon_max), latitude = slice(self.lat_min, self.lat_max), drop = True).sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
        
        self.N_lon = self.ds_model.dims['longitude']
        self.N_lat = self.ds_model.dims['latitude']
        self.N_time = self.ds_model.dims['time']

        if self.N_lat != input_dim[0]:
            raise Exception(f'Incompatible lon/lat bounds: N_lon = {self.N_lon} but should be {self.input_dim[1]}, N_lat = {self.N_lat} but should be {self.input_dim[0]}')
        if self.N_lon != input_dim[1]:
            raise Exception(f'Incompatible lon/lat bounds: N_lon = {self.N_lon} but should be {self.input_dim[1]}, N_lat = {self.N_lat} but should be {self.input_dim[0]}')    
        
    def __len__(self):
        return int((self.date_range[1]-self.date_range[0]).days)
    
    def worker_init_fn(self, worker_id):
        self.ds_model = xr.open_dataset(self.data_dir + self.model_file)
        ds_m = xr.open_dataset(self.data_dir + self.mean_file)
        monthly_climatology = xr.open_dataset(self.data_dir + self.clim_file)
        
        if 'so' in self.variables:
            self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']
        if 'thetao' in self.variables:
            self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']
        
        for var in [v for v in self.variables if v not in ['so','thetao']]: 
            self.ds_model[var] = self.ds_model[var] - ds_m[var]
            
        self.ds_model = self.ds_model.isel(depth=0,drop=True).sel(longitude = slice(self.lon_min, self.lon_max), latitude = slice(self.lat_min, self.lat_max), drop = True).sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
        
        self.N_lon = self.ds_model.dims['longitude']
        self.N_lat = self.ds_model.dims['latitude']
        self.N_time = self.ds_model.dims['time']

    def __getitem__(self, idx):

        invar = None
        
        data_model = self.ds_model.isel(time = idx)
        
        outvar = np.zeros((self.n_channels, self.input_dim[0], self.input_dim[1]))
        for v, var in enumerate(self.variables):
            outvar[v,] = data_model[var]/self.var_stds[var]
        
        outvar = torch.from_numpy(outvar.astype(np.float32))
        outvar = torch.nan_to_num(outvar, nan=0.0)
        
        return invar, outvar
    
