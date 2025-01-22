import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from datetime import date, timedelta
import numpy as np

class Diffusion_Training_Dataset(Dataset):
    """
    A PyTorch Dataset for multi-modal diffusion training using GLORYS ocean data.

    This dataset is designed for random sampling of spatio-temporal patches of oceanographic variables
    for use in training generative models. Data is dynamically loaded and preprocessed, supporting 
    multiprocessing and optional data augmentation.

    Attributes:
        data_dir (str): Path to the directory containing the input GLORYS dataset files.
        n_lon (int): Number of longitude pixels in the reconstruction patch.
        n_lat (int): Number of latitude pixels in the reconstruction patch.
        date_range (list): A list of two `datetime` objects specifying the start and end dates for sampling.
        variables (list): Names of variables to extract from the dataset, e.g., ['zos', 'so', 'thetao', 'uo', 'vo'].
        var_stds (dict): Dictionary mapping variable names to their standard deviations for normalization.
        lon_buffers (list): List of two integers specifying the number of longitude pixels to exclude from each edge. 
                            Use `None` for no buffer (default: [None, None]).
        lat_buffers (list): List of two integers specifying the number of latitude pixels to exclude from each edge. 
                            Use `None` for no buffer (default: [None, None]).
        model_file (str): Filename of the main GLORYS dataset file (default: 'glorys_pre_processed_fixed_noislands.nc').
        mean_file (str): Filename of the mean values file used for normalization 
                         (default: 'glorys_means_pre_processed_fixed_noislands.nc').
        clim_file (str): Filename of the climatology file used for seasonal adjustment 
                         (default: 'glorys_gulfstream_climatology.nc').
        multiprocessing (bool): Whether to use multiprocessing for data loading (default: True).
        augment (bool): Whether to apply data augmentation by adding random offsets to selected variables 
                        (default: False).

    Methods:
        __len__(): 
            Returns an arbitrary large number, allowing infinite sampling for diffusion training.
        worker_init_fn(worker_id): 
            Initializes dataset resources for multiprocessing workers.
        __getitem__(idx): 
            Generates a random spatio-temporal patch of data.

            Returns:
                torch.Tensor: A tensor of shape (len(variables), n_lat, n_lon) with normalized data for the requested variables.
                The tensor is filled with zero if `nan` values are present in the data.
    """
    def __init__(self, 
                 data_dir, 
                 n_lon, 
                 n_lat, 
                 date_range, 
                 variables, 
                 var_stds, 
                 lon_buffers = [None, None], 
                 lat_buffers = [None, None], 
                 model_file = 'glorys_pre_processed_fixed_noislands.nc', 
                 mean_file = 'glorys_means_pre_processed_fixed_noislands.nc', 
                 clim_file = 'glorys_gulfstream_climatology.nc', 
                 multiprocessing = True, 
                 augment = False,
                ):
        self.data_dir = data_dir
        self.n_lon = n_lon
        self.n_lat = n_lat
        self.lon_buffers = lon_buffers
        self.lat_buffers = lat_buffers
        self.date_range = date_range
        self.variables = variables
        self.var_stds = var_stds
        self.n_channels = len(variables)
        self.model_file = model_file
        self.mean_file = mean_file
        self.clim_file = clim_file
        self.multiprocessing = multiprocessing
        
        if not multiprocessing:
            self.ds_model = xr.open_dataset(self.data_dir + self.model_file)
            ds_m = xr.open_dataset(self.data_dir + self.mean_file)
            monthly_climatology = xr.open_dataset(self.data_dir + self.clim_file).isel(depth = 0, drop = True)

            if 'so' in self.variables:
                self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']
            if 'thetao' in self.variables:
                self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']

            for var in [v for v in self.variables if v not in ['so','thetao']]:   #['zos','uo','vo']:
                self.ds_model[var] = self.ds_model[var] - ds_m[var]

            self.ds_model = self.ds_model.drop('month')#.isel(depth = 0, drop = True)

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

            self.ds_model = self.ds_model.isel(longitude = slice(i_lon_min, i_lon_max), latitude = slice(i_lat_min, i_lat_max))

            self.N_lon = self.ds_model.dims['longitude']
            self.N_lat = self.ds_model.dims['latitude']

            self.ds_model = self.ds_model.sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
            self.N_time = self.ds_model.dims['time']
        
    def __len__(self):
        return int(1e9) # some arbitrary large number. The sampling operation just selects a random spatio-temporal crop from the larger dataset so this number should just be > number of training steps.
    
    def worker_init_fn(self, worker_id):
        # initialize dataset on each worker if multi-processing used.
        self.ds_model = xr.open_dataset(self.data_dir + self.model_file)
        ds_m = xr.open_dataset(self.data_dir + self.mean_file)
        monthly_climatology = xr.open_dataset(self.data_dir + self.clim_file).isel(depth = 0, drop = True)
        
        if 'so' in self.variables:
            self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']
        if 'thetao' in self.variables:
            self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']

        for var in [v for v in self.variables if v not in ['so','thetao']]:   #['zos','uo','vo']:
            self.ds_model[var] = self.ds_model[var] - ds_m[var]
        
        self.ds_model = self.ds_model.drop('month')#.isel(depth = 0, drop = True)
        
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
            
        self.ds_model = self.ds_model.isel(longitude = slice(i_lon_min, i_lon_max), latitude = slice(i_lat_min, i_lat_max))

        self.N_lon = self.ds_model.dims['longitude']
        self.N_lat = self.ds_model.dims['latitude']
        
        self.ds_model = self.ds_model.sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
        self.N_time = self.ds_model.dims['time']
        
    def __getitem__(self, idx):
        still_checking = True
        while still_checking:
           # completely randomised (i.e. getitem independent of idx!!) sampling to allow infinite sample generation for diffusion training: 
            lat_start = np.random.randint(self.N_lat - self.n_lat + 1)
            lon_start = np.random.randint(self.N_lon - self.n_lon + 1)
            indexer = {
                       'latitude': slice(lat_start, lat_start + self.n_lat),
                       'longitude': slice(lon_start, lon_start + self.n_lon), 
                       'time': np.random.randint(self.N_time)
                      }
            
            data_model = self.ds_model.isel(indexer, drop = True)
            # data_oi = self.ds_oi.isel(indexer, drop = True)

            outvar = np.zeros((self.n_channels, self.n_lat, self.n_lon))
            for v, var in enumerate(self.variables):
                outvar[v,] = data_model[var]
                outvar[v,] /= self.var_stds[var]
                if self.augment: # optional data augmentation to improve robustness to large-scale SSH/SST/SSS variability, add random constant offset to each field during training
                    if var == 'zos':
                        outvar[v, ] += (np.random.uniform(1) - 0.5) * 0.2
                    if var == 'thetao':
                        outvar[v, ] += (np.random.uniform(1) - 0.5) * 0.2
                    if var == 'so':
                        outvar[v, ] += (np.random.uniform(1) - 0.5) * 0.2

            if np.size(outvar[np.isnan(outvar)]) == 0:
                outvar = torch.from_numpy(outvar.astype(np.float32))
                still_checking = False
                outvar = torch.nan_to_num(outvar, nan=0.0)
                
        return outvar
    
# difference from training dataset: this one takes fixed test region and draws sequentially in time.
class GenDA_OSSE_Inference_Dataset(Dataset):
    """
    A PyTorch Dataset for multi-modal generative data assimilation inference using GLORYS data.
    The dataset samples daily in order from a fixed sub-domain of the dataset.

    Attributes:
        data_dir (str): Directory containing all input data files.
        lon_min (float): Minimum longitude of the reconstruction domain.
        lon_max (float): Maximum longitude of the reconstruction domain.
        lat_min (float): Minimum latitude of the reconstruction domain.
        lat_max (float): Maximum latitude of the reconstruction domain.
        input_dim (int or tuple of ints): Size of the neural network reconstruction patch.
            If an int, the patch is square (N, N). If a tuple, it specifies (N_lat, N_lon).
        date_range (list): List of two datetime objects defining the sampling period (start and end dates).
        variables (list): List of variable names to extract from the dataset (e.g., 'zos', 'so', 'thetao', 'uo', 'vo').
        var_stds (dict): Dictionary of standard deviations for normalizing each variable in `variables`.
        model_file (str): Filename for the main model dataset (default: 'glorys_pre_processed_fixed_noislands.nc').
        mean_file (str): Filename for the dataset of mean climatology values (default: 'glorys_means_pre_processed_fixed_noislands.nc').
        clim_file (str): Filename for the dataset of monthly climatology values (default: 'glorys_gulfstream_climatology.nc').
        multiprocessing (bool): Indicates if multi-processing will be used in the DataLoader. 
            Required for worker initialization (default: False).

    Methods:
        __len__():
            Returns the number of time samples (days) in the dataset.

        __getitem__(idx):
            Retrieves a single sample from the dataset for the specified index.

            Returns:
                outvar (torch.Tensor): Output tensor of shape 
                    (n_channels, input_dim[0], input_dim[1]), where
                    n_channels is the number of variables,
                    and input_dim[0], input_dim[1] are the latitudinal and longitudinal patch dimensions, respectively.
    """
    def __init__(self, 
                 data_dir, 
                 lon_min, 
                 lon_max, 
                 lat_min, 
                 lat_max, 
                 input_dim, 
                 date_range, 
                 variables, 
                 var_stds,
                 model_file = 'glorys_pre_processed_fixed_noislands.nc', 
                 mean_file = 'glorys_means_pre_processed_fixed_noislands.nc', 
                 clim_file = 'glorys_gulfstream_climatology.nc',
                 multiprocessing = False
                ):
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
        self.model_file = model_file
        self.mean_file = mean_file
        self.clim_file = clim_file

        self.ds_model = xr.open_dataset(self.data_dir + self.model_file).astype('float32') 
        self.ds_m = xr.open_dataset(self.data_dir + self.mean_file).astype('float32')
        self.monthly_climatology = xr.open_dataset(self.data_dir + self.clim_file).astype('float32')

        if 'so' in self.variables:
            self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - self.monthly_climatology['thetao']
        if 'thetao' in self.variables:
            self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - self.monthly_climatology['so']

        for var in [v for v in self.variables if v not in ['so','thetao']]:   #['zos','uo','vo']:
            self.ds_model[var] = self.ds_model[var] - self.ds_m[var]
        
        self.ds_model = self.ds_model.drop('month')#.isel(depth = 0, drop = True)

        self.ds_model = self.ds_model.sel(longitude = slice(self.lon_min, self.lon_max), latitude = slice(self.lat_min, self.lat_max), drop = True).sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)

        self.ds_m = self.ds_m.sel(longitude = slice(self.lon_min, self.lon_max), latitude = slice(self.lat_min, self.lat_max), drop = True)

        self.monthly_climatology = self.monthly_climatology.sel(longitude = slice(self.lon_min, self.lon_max), latitude = slice(self.lat_min, self.lat_max), drop = True)
        
        self.N_lon = self.ds_model.dims['longitude']
        self.N_lat = self.ds_model.dims['latitude']
        self.N_time = self.ds_model.dims['time']

        if self.N_lat != input_dim[0]:
            raise Exception(f'Incompatible lon/lat bounds: N_lon = {self.N_lon} but should be {self.input_dim[1]}, N_lat = {self.N_lat} but should be {self.input_dim[0]}')
        if self.N_lon != input_dim[1]:
            raise Exception(f'Incompatible lon/lat bounds: N_lon = {self.N_lon} but should be {self.input_dim[1]}, N_lat = {self.N_lat} but should be {self.input_dim[0]}')    
        
    def __len__(self):
        return int(self.N_time)
        
    def __getitem__(self, idx):
        
        data_model = self.ds_model.isel(time = idx)
        
        outvar = np.zeros((self.n_channels, self.input_dim[0], self.input_dim[1]))
        for v, var in enumerate(self.variables):
            outvar[v,] = data_model[var]/self.var_stds[var]
        
        outvar = torch.from_numpy(outvar.astype(np.float32))
        outvar = torch.nan_to_num(outvar, nan=0.0)
        
        return outvar
    
class L3L4_Regression_Training_Dataset(Dataset):
    """
    A PyTorch Dataset for training CNNs to estimate multi-modal surface ocean states 
    using L3 observations and L4 OI products. The dataset is designed for comparison 
    with generative data assimilation techniques.

    Attributes:
        data_dir (str): Path to the directory containing all input data files.
        n_lon (int): Number of longitude pixels in the reconstruction patch.
        n_lat (int): Number of latitude pixels in the reconstruction patch.
        date_range (list): List of two `datetime` objects defining the sampling period.
        variables_in (list): Names of input variables (e.g., ['thetao', 'zos']).
        variables_oi (list): Names of observational intermediate variables.
        variables_out (list): Names of target output variables.
        var_stds (dict): Mapping of variable names to their standard deviations for normalization.
        model_file (str): Filename of the GLORYS dataset (default: 'glorys_pre_processed_fixed_noislands.nc').
        mean_file (str): Filename of the mean values file for normalization 
                         (default: 'glorys_means_pre_processed_fixed_noislands.nc').
        clim_file (str): Filename of the climatology file for seasonal adjustments 
                         (default: 'glorys_gulfstream_climatology.nc').
        oi_file (str): Filename of the L4 OI product file 
                       (default: 'oi_l4_ssh-sst-sss_full_domain_updated_with_errors_sigmas_ssh25_sst16_sss16_norescaling_nodemean_noislands.nc').
        mask_file (str): Filename of the observation mask file (default: 'obs_masks_ssh-sst-u-v.nc').
        noise_file (str): Filename of the noise dataset file (default: 'OSE_L3_products.nc').
        lon_buffers (list): Longitude buffer sizes to exclude from the sampling edges, as [min, max] 
                            (default: [None, None]).
        lat_buffers (list): Latitude buffer sizes to exclude from the sampling edges, as [min, max] 
                            (default: [None, None]).
        multiprocessing (bool): Indicates if multi-processing is used for data loading (default: True).
        add_obs_noise (bool): Whether to add observational noise to the input data (default: False).
        noise_attenuation_factor (float): Factor to scale the added noise (default: 0.2).

    Methods:
        __len__(): 
            Returns a very large number to allow virtually unlimited sampling for training.
        __getitem__(idx): 
            Generates input-output data pairs for training.

            Returns:
                invar (torch.Tensor): Tensor of shape (n_channels_in + n_channels_oi, n_lat, n_lon) 
                                      with normalized input data.
                outvar (torch.Tensor): Tensor of shape (n_channels_out, n_lat, n_lon) with normalized 
                                       target data.
        worker_init_fn(worker_id): 
            Initializes dataset resources for multiprocessing workers.
    """
    def __init__(self, 
                 data_dir, 
                 n_lon, 
                 n_lat, 
                 date_range, 
                 variables_in, 
                 variables_oi, 
                 variables_out, 
                 var_stds, 
                 model_file = 'glorys_pre_processed_fixed_noislands.nc', 
                 mean_file = 'glorys_means_pre_processed_fixed_noislands.nc', 
                 clim_file = 'glorys_gulfstream_climatology.nc', 
                 oi_file = 'oi_l4_ssh-sst-sss_full_domain_updated_with_errors_sigmas_ssh25_sst16_sss16_norescaling_nodemean_noislands.nc', 
                 mask_file = 'obs_masks_ssh-sst-u-v.nc', 
                 noise_file = 'OSE_L3_products.nc',
                 lon_buffers = [None, None], 
                 lat_buffers = [None, None], 
                 multiprocessing = True, 
                 add_obs_noise = False, 
                 noise_attenuation_factor = 0.2,
                ):
        self.data_dir = data_dir
        self.n_lon = n_lon
        self.n_lat = n_lat
        self.lon_buffers = lon_buffers
        self.lat_buffers = lat_buffers
        self.date_range = date_range
        self.variables_in = variables_in
        self.variables_oi = variables_oi
        self.variables_out = variables_out
        self.var_stds = var_stds
        self.n_channels_in = len(variables_in)
        self.n_channels_oi = len(variables_oi)
        self.n_channels_out = len(variables_out)
        self.model_file = model_file
        self.mean_file = mean_file
        self.clim_file = clim_file
        self.oi_file = oi_file
        self.mask_file = mask_file
        self.noise_file = noise_file
        self.multiprocessing = multiprocessing
        self.add_obs_noise = add_obs_noise
        self.noise_attenuation_factor = noise_attenuation_factor
        
        self.ds_model = xr.open_dataset(self.data_dir + self.model_file).astype('float32')
        ds_m = xr.open_dataset(self.data_dir + self.mean_file).astype('float32')
        monthly_climatology = xr.open_dataset(self.data_dir + self.clim_file).astype('float32')
        self.ds_oi = xr.open_dataset(self.data_dir + self.oi_file).astype('float32')

        self.ds_masks = xr.open_dataset(self.data_dir + self.mask_file)
        self.ds_masks['ssh_mask'] = (self.ds_masks['ssh_nadir'].astype('bool')) | (self.ds_masks['ssh_karin'].astype('bool'))

        if ('so' in self.variables_in) or ('so' in self.variables_out):
            self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']
            
            
        if ('thetao' in self.variables_in) or ('thetao' in self.variables_out):
            self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']

        for var in [v for v in unique_elements(self.variables_in, self.variables_out) if v not in ['thetao', 'so']]:
            self.ds_model[var] = self.ds_model[var] - ds_m[var]

        self.ds_oi['ssh_oi'] = self.ds_oi['ssh_oi'] - ds_m['zos']
        self.ds_oi['sst_oi'] = self.ds_oi['sst_oi'].groupby('time.month') - monthly_climatology['thetao']
        self.ds_oi['sss_oi'] = self.ds_oi['sss_oi'].groupby('time.month') - monthly_climatology['so']
        
        self.ds_oi['ssh_oi'] = self.ds_oi['ssh_oi']/var_stds['zos']
        self.ds_oi['sst_oi'] = self.ds_oi['sst_oi']/var_stds['thetao']
        self.ds_oi['sss_oi'] = self.ds_oi['sss_oi']/var_stds['so']
        self.ds_oi['ssh_oi_standard_error'] = self.ds_oi['ssh_oi_standard_error']/var_stds['zos']
        self.ds_oi['sst_oi_standard_error'] = self.ds_oi['sst_oi_standard_error']/var_stds['thetao']
        self.ds_oi['sss_oi_standard_error'] = self.ds_oi['sss_oi_standard_error']/var_stds['so']
        
        self.ds_model = self.ds_model.drop('month')

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
            
        self.ds_model = self.ds_model.isel(longitude = slice(i_lon_min, i_lon_max), latitude = slice(i_lat_min, i_lat_max))
        self.ds_oi = self.ds_oi.isel(longitude = slice(i_lon_min, i_lon_max), latitude = slice(i_lat_min, i_lat_max))
        
        self.N_lon = self.ds_model.dims['longitude']
        self.N_lat = self.ds_model.dims['latitude']
        
        self.ds_model = self.ds_model.sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
        self.ds_oi = self.ds_oi.sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
        self.N_time = self.ds_model.dims['time']

        ds_noise = xr.open_dataset(self.data_dir + self.noise_file)

        self.ssh_obs_noise_std = float(ds_noise['ssh_error'].mean())
        self.sst_obs_noise_std = float(ds_noise['sst_error'].mean())
 
    def __len__(self):
        return int(1e9)

    def __getitem__(self, idx):
        still_checking = True
        while still_checking:

            lat_start = np.random.randint(self.N_lat - self.n_lat + 1)
            lon_start = np.random.randint(self.N_lon - self.n_lon + 1)
            indexer = {
                       'latitude': slice(lat_start, lat_start + self.n_lat),
                       'longitude': slice(lon_start, lon_start + self.n_lon), 
                       'time': np.random.randint(self.N_time)
                      }
            
            data_model = self.ds_model.isel(indexer)
            data_oi = self.ds_oi.isel(indexer)

            invar = np.zeros((self.n_channels_in + self.n_channels_oi, self.n_lat, self.n_lon))
            outvar = np.zeros((self.n_channels_out, self.n_lat, self.n_lon))

            sst_mask_idx = np.random.randint(365)
            ssh_mask_idx = np.random.randint(365)

            self.ds_masks['longitude'] = data_model['longitude']
            self.ds_masks['latitude'] = data_model['latitude']

            for v, var in enumerate(self.variables_in):
                if var == 'thetao':
                    if self.add_obs_noise:
                        if self.noise_attenuation_factor is not None:
                            invar[v,] = self.ds_masks['sst_mask'].isel(time = sst_mask_idx) * ((data_model[var]/self.var_stds[var]) + self.noise_attenuation_factor * self.sst_obs_noise_std * np.random.randn(self.n_lat, self.n_lon))
                        else:
                            invar[v,] = self.ds_masks['sst_mask'].isel(time = sst_mask_idx) * ((data_model[var]/self.var_stds[var]) + self.sst_obs_noise_std * np.random.randn(self.n_lat, self.n_lon))
                    else:
                        invar[v,] = (data_model[var]/self.var_stds[var])*self.ds_masks['sst_mask'].isel(time = sst_mask_idx)
                
                elif var == 'zos':
                    if self.add_obs_noise:
                        if self.noise_attenuation_factor is not None:
                            invar[v,] = self.ds_masks['ssh_mask'].isel(time = ssh_mask_idx) * ((data_model[var]/self.var_stds[var]) + self.noise_attenuation_factor * self.ssh_obs_noise_std * np.random.randn(self.n_lat, self.n_lon))
                        else:
                            invar[v,] = self.ds_masks['ssh_mask'].isel(time = ssh_mask_idx) * ((data_model[var]/self.var_stds[var]) + self.ssh_obs_noise_std * np.random.randn(self.n_lat, self.n_lon))
                    else:
                        invar[v,] = (data_model[var]/self.var_stds[var])*self.ds_masks['ssh_mask'].isel(time = ssh_mask_idx)
                
                else:
                    invar[v,] = (data_model[var]/self.var_stds[var])
            
            for v, var in enumerate(self.variables_oi):
                invar[v + self.n_channels_in,] = data_oi[var]#/self.oi_stds[var]
                
            for v, var in enumerate(self.variables_out):
                outvar[v,] = data_model[var]/self.var_stds[var]

            if np.size(outvar[np.isnan(outvar)]) == 0:
                outvar = torch.from_numpy(outvar.astype(np.float32))
                invar = torch.from_numpy(invar.astype(np.float32))
                still_checking = False
                outvar = torch.nan_to_num(outvar, nan=0.0)
                invar = torch.nan_to_num(invar, nan=0.0)
        
        return invar, outvar
    
class L3L4_Regression_OSSE_Inference_Dataset(Dataset):
    """
    A PyTorch Dataset designed for training convolutional neural networks (CNNs) 
    to estimate the multi-modal surface ocean state from Level 3 (L3) observations 
    and Level 4 (L4) objective interpolation (OI) products, as part of an 
    Observing System Simulation Experiment (OSSE). This dataset supports comparison 
    with generative data assimilation methods.

    Attributes:
        data_dir (str): Path to the directory containing input data files.
        lon_min (float): Minimum longitude bound for the reconstruction domain.
        lon_max (float): Maximum longitude bound for the reconstruction domain.
        lat_min (float): Minimum latitude bound for the reconstruction domain.
        lat_max (float): Maximum latitude bound for the reconstruction domain.
        input_dim (int): Dimensionality of the input data.
        date_range (list): A list of two `datetime` objects representing the start 
            and end dates of the sampling period.
        variables_in (list): Names of variables used as input to the model.
        variables_oi (list): Names of OI variables to include as input.
        variables_out (list): Names of target variables for model output.
        var_stds (dict): Standard deviations for normalizing each variable.
        model_file (str): Name of the NetCDF file containing model data.
        mean_file (str): Name of the file containing mean state data for normalization.
        clim_file (str): Name of the file containing climatology data.
        oi_file (str): Name of the file containing L4 OI data.
        mask_file (str): Name of the file containing observational masks.
        noise_file (str): Name of the file containing observation noise data.
        add_obs_noise (bool): Whether to add observational noise to input variables.
        noise_attenuation_factor (float or None): Factor to scale added noise; 
            if `None`, defaults to the standard noise level.
        multiprocessing (bool): Indicates if multiprocessing is used during 
            dataset initialization. Must be `True` (default behavior).

    Methods:
        __len__():
            Returns the total number of time steps (samples) in the dataset, 
            based on the sampling period defined by `date_range`.
        
        worker_init_fn(worker_id):
            Initializes worker processes for multiprocessing.

        __getitem__(idx):
            Retrieves input-output data pairs for the specified index.
            
            Returns:
                invar (torch.Tensor): A tensor of shape 
                    (len(variables_in) + len(variables_oi), lat, lon) containing 
                    normalized input data. Noise may be added to specific variables 
                    if `add_obs_noise` is True.
                outvar (torch.Tensor): A tensor of shape 
                    (len(variables_out), lat, lon) containing normalized target data.
    """
    def __init__(self, 
                 data_dir, 
                 lon_min, 
                 lon_max, 
                 lat_min, 
                 lat_max, 
                 input_dim, 
                 date_range, 
                 variables_in, 
                 variables_oi, 
                 variables_out, 
                 var_stds,
                 model_file = 'glorys_pre_processed_fixed_noislands.nc', 
                 mean_file = 'glorys_means_pre_processed_fixed_noislands.nc', 
                 clim_file = 'glorys_gulfstream_climatology.nc', 
                 oi_file = 'oi_l4_ssh-sst-sss_full_domain_updated_with_errors_sigmas_ssh25_sst16_sss16_norescaling_nodemean_noislands.nc', 
                 mask_file = 'obs_masks_ssh-sst-u-v.nc', 
                 noise_file = 'OSE_L3_products.nc',
                 multiprocessing = True, 
                 add_obs_noise = False, 
                 noise_attenuation_factor = None
                ):
        self.data_dir = data_dir
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.input_dim = input_dim
        self.date_range = date_range
        self.variables_in = variables_in
        self.variables_oi = variables_oi
        self.variables_out = variables_out
        self.n_channels_in = len(variables_in)
        self.n_channels_oi = len(variables_oi)
        self.n_channels_out = len(variables_out)
        self.var_stds = var_stds
        self.model_file = model_file
        self.mean_file = mean_file
        self.clim_file = clim_file
        self.oi_file = oi_file
        self.mask_file = mask_file
        self.noise_file = noise_file
        self.add_obs_noise = add_obs_noise
        self.noise_attenuation_factor = noise_attenuation_factor
        
        if not multiprocessing:
        
            self.ds_model = xr.open_dataset(self.data_dir + self.model_file).astype('float32')
            ds_m = xr.open_dataset(self.data_dir + self.mean_file).astype('float32')
            monthly_climatology = xr.open_dataset(self.data_dir + self.clim_file).astype('float32')
            self.ds_oi = xr.open_dataset(self.data_dir + self.oi_file).astype('float32')

            self.ds_masks = xr.open_dataset(self.data_dir + self.mask_file)
            self.ds_masks['ssh_mask'] = (self.ds_masks['ssh_nadir'].astype('bool')) | (self.ds_masks['ssh_karin'].astype('bool'))

            if ('so' in self.variables_in) or ('so' in self.variables_out):
                self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']


            if ('thetao' in self.variables_in) or ('thetao' in self.variables_out):
                self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']

            for var in [v for v in unique_elements(self.variables_in, self.variables_out) if v not in ['thetao', 'so']]:
                self.ds_model[var] = self.ds_model[var] - ds_m[var]

            self.ds_oi['ssh_oi'] = self.ds_oi['ssh_oi'] - ds_m['zos']
            self.ds_oi['sst_oi'] = self.ds_oi['sst_oi'].groupby('time.month') - monthly_climatology['thetao']
            self.ds_oi['sss_oi'] = self.ds_oi['sss_oi'].groupby('time.month') - monthly_climatology['so']

            self.ds_oi['ssh_oi'] = self.ds_oi['ssh_oi']/var_stds['zos']
            self.ds_oi['sst_oi'] = self.ds_oi['sst_oi']/var_stds['thetao']
            self.ds_oi['sss_oi'] = self.ds_oi['sss_oi']/var_stds['so']
            self.ds_oi['ssh_oi_standard_error'] = self.ds_oi['ssh_oi_standard_error']/var_stds['zos']
            self.ds_oi['sst_oi_standard_error'] = self.ds_oi['sst_oi_standard_error']/var_stds['thetao']
            self.ds_oi['sss_oi_standard_error'] = self.ds_oi['sss_oi_standard_error']/var_stds['so']

            self.ds_model = self.ds_model.drop('month')

            self.ds_model = self.ds_model.sel(longitude = slice(lon_min, lon_max), latitude = slice(lat_min, lat_max))
            self.ds_oi = self.ds_oi.sel(longitude = slice(lon_min, lon_max), latitude = slice(lat_min, lat_max))

            self.N_lon = self.ds_model.dims['longitude']
            self.N_lat = self.ds_model.dims['latitude']

            self.ds_model = self.ds_model.sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
            self.ds_oi = self.ds_oi.sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
            self.N_time = self.ds_model.dims['time']

            ds_noise = xr.open_dataset(self.data_dir + self.noise_file)

            self.ssh_obs_noise_std = float(ds_noise['ssh_error'].mean())
            self.sst_obs_noise_std = float(ds_noise['sst_error'].mean())
        
        else:
            raise Exception("multiprocessing not implemented")

    def __len__(self):
        return int((self.date_range[1]-self.date_range[0]).days)
    
    def __getitem__(self, idx):
       
        data_model = self.ds_model.isel(time = idx)
        data_oi = self.ds_oi.isel(time = idx)

        invar = np.zeros((self.n_channels_in + self.n_channels_oi, self.N_lat, self.N_lon))
        outvar = np.zeros((self.n_channels_out, self.N_lat, self.N_lon))

        self.ds_masks['longitude'] = data_model['longitude']
        self.ds_masks['latitude'] = data_model['latitude']

        for v, var in enumerate(self.variables_in):
            if var == 'thetao':
                if self.add_obs_noise:
                    if self.noise_attenuation_factor is not None:
                        invar[v,] = self.ds_masks['sst_mask'].isel(time = idx) * ((data_model[var]/self.var_stds[var]) + self.noise_attenuation_factor * self.sst_obs_noise_std * np.random.randn(self.N_lat, self.N_lon))
                    else:
                        invar[v,] = self.ds_masks['sst_mask'].isel(time = idx) * ((data_model[var]/self.var_stds[var]) + self.sst_obs_noise_std * np.random.randn(self.N_lat, self.N_lon))
                else:
                    invar[v,] = (data_model[var]/self.var_stds[var])*self.ds_masks['sst_mask'].isel(time = idx)
            
            elif var == 'zos':
                if self.add_obs_noise:
                    if self.noise_attenuation_factor is not None:
                        invar[v,] = self.ds_masks['ssh_mask'].isel(time = idx) * ((data_model[var]/self.var_stds[var]) + self.noise_attenuation_factor * self.ssh_obs_noise_std * np.random.randn(self.N_lat, self.N_lon))
                    else:
                        invar[v,] = self.ds_masks['ssh_mask'].isel(time = idx) * ((data_model[var]/self.var_stds[var]) + self.ssh_obs_noise_std * np.random.randn(self.N_lat, self.N_lon))
                else:
                    invar[v,] = (data_model[var]/self.var_stds[var])*self.ds_masks['ssh_mask'].isel(time = idx)
            
            else:
                invar[v,] = (data_model[var]/self.var_stds[var])
        
        for v, var in enumerate(self.variables_oi):
            invar[v + self.n_channels_in,] = data_oi[var]
            
        for v, var in enumerate(self.variables_out):
            outvar[v,] = data_model[var]/self.var_stds[var]

        invar = torch.from_numpy(invar.astype(np.float32))
        invar = torch.nan_to_num(invar, nan=0.0)
        
        outvar = torch.from_numpy(outvar.astype(np.float32))
        outvar = torch.nan_to_num(outvar, nan=0.0)
        
        return invar, outvar
    
class L3L4_Regression_OSE_Inference_Dataset(Dataset):
    """
    A PyTorch Dataset designed for training convolutional neural networks (CNNs) 
    to estimate the multi-modal surface ocean state from Level 3 (L3) observations 
    and Level 4 (L4) objective interpolation (OI) products, as part of an 
    Observing System Simulation Experiment (OSSE). This dataset supports comparison 
    with generative data assimilation methods.

    Attributes:
        data_dir (str): Path to the directory containing input data files.
        lon_min (float): Minimum longitude bound for the reconstruction domain.
        lon_max (float): Maximum longitude bound for the reconstruction domain.
        lat_min (float): Minimum latitude bound for the reconstruction domain.
        lat_max (float): Maximum latitude bound for the reconstruction domain.
        input_dim (int): Dimensionality of the input data.
        date_range (list): A list of two `datetime` objects representing the start 
            and end dates of the sampling period.
        variables_in (list): Names of variables used as input to the model.
        variables_oi (list): Names of OI variables to include as input.
        variables_out (list): Names of target variables for model output.
        var_stds (dict): Standard deviations for normalizing each variable.
        model_file (str): Name of the NetCDF file containing model data.
        mean_file (str): Name of the file containing mean state data for normalization.
        clim_file (str): Name of the file containing climatology data.
        oi_file (str): Name of the file containing L4 OI data.
        mask_file (str): Name of the file containing observational masks.
        noise_file (str): Name of the file containing observation noise data.
        add_obs_noise (bool): Whether to add observational noise to input variables.
        noise_attenuation_factor (float or None): Factor to scale added noise; 
            if `None`, defaults to the standard noise level.
        multiprocessing (bool): Indicates if multiprocessing is used during 
            dataset initialization. Must be `True` (default behavior).

    Methods:
        __len__():
            Returns the total number of time steps (samples) in the dataset, 
            based on the sampling period defined by `date_range`.
        
        worker_init_fn(worker_id):
            Initializes worker processes for multiprocessing.

        __getitem__(idx):
            Retrieves input-output data pairs for the specified index.
            
            Returns:
                invar (torch.Tensor): A tensor of shape 
                    (len(variables_in) + len(variables_oi), lat, lon) containing 
                    normalized input data. Noise may be added to specific variables 
                    if `add_obs_noise` is True.
                outvar (torch.Tensor): A tensor of shape 
                    (len(variables_out), lat, lon) containing normalized target data.
    """
    def __init__(self, 
                 data_dir, 
                 lon_min, 
                 lon_max, 
                 lat_min, 
                 lat_max, 
                 input_dim, 
                 date_range, 
                 variables_in, 
                 variables_oi, 
                 variables_out, 
                 var_stds, 
                 model_file = 'glorys_pre_processed_fixed_noislands.nc', 
                 mean_file = 'glorys_means_pre_processed_fixed_noislands.nc', 
                 clim_file = 'glorys_gulfstream_climatology.nc', 
                 oi_file = 'OSE_L4_products_v2.nc', 
                 obs_file = 'OSE_L3_products_v2.nc',
                 mask_file = 'obs_masks_ssh-sst-u-v.nc',
                 multiprocessing = True, 
                 double_sst_mask = False
                ):
        self.data_dir = data_dir
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.input_dim = input_dim
        self.date_range = date_range
        self.variables_in = variables_in
        self.variables_oi = variables_oi
        self.variables_out = variables_out
        self.n_channels_in = len(variables_in)
        self.n_channels_oi = len(variables_oi)
        self.n_channels_out = len(variables_out)
        self.var_stds = var_stds
        self.model_file = model_file
        self.mean_file = mean_file
        self.clim_file = clim_file
        self.oi_file = oi_file
        self.mask_file = mask_file
        self.obs_file = obs_file
        self.double_sst_mask = double_sst_mask
        
        self.ds_model = xr.open_dataset(self.data_dir + self.model_file).astype('float32')
        ds_m = xr.open_dataset(self.data_dir + self.mean_file).astype('float32')
        monthly_climatology = xr.open_dataset(self.data_dir + self.clim_file).astype('float32')

        self.ds_obs = xr.open_dataset(self.data_dir + self.obs_file)
        self.ds_oi = xr.open_dataset(self.data_dir + self.oi_file)

        self.ds_oi = self.ds_oi.transpose('time','latitude','longitude')
        self.ds_obs = self.ds_obs.transpose('time','latitude','longitude')

        self.ds_masks = xr.open_dataset(self.data_dir + self.mask_file)
        self.ds_masks['ssh_mask'] = (self.ds_masks['ssh_nadir'].astype('bool')) | (self.ds_masks['ssh_karin'].astype('bool'))

        if ('so' in self.variables_in) or ('so' in self.variables_out):
            self.ds_model['so'] = self.ds_model['so'].groupby('time.month') - monthly_climatology['so']
            
            
        if ('thetao' in self.variables_in) or ('thetao' in self.variables_out):
            self.ds_model['thetao'] = self.ds_model['thetao'].groupby('time.month') - monthly_climatology['thetao']

        for var in [v for v in unique_elements(self.variables_in, self.variables_out) if v not in ['thetao', 'so']]:
            self.ds_model[var] = self.ds_model[var] - ds_m[var]
        
        self.ds_model = self.ds_model.drop('month')
    
        self.ds_model = self.ds_model.sel(longitude = slice(lon_min, lon_max), latitude = slice(lat_min, lat_max))
        self.ds_oi = self.ds_oi.sel(longitude = slice(lon_min, lon_max), latitude = slice(lat_min, lat_max))
        
        self.N_lon = self.ds_model.dims['longitude']
        self.N_lat = self.ds_model.dims['latitude']

        self.ds_model = self.ds_model.sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
        self.ds_oi = self.ds_oi.sel(time=slice(self.date_range[0],self.date_range[1]), drop = True)
        self.N_time = self.ds_model.dims['time']
 
    def __len__(self):
        return int((self.date_range[1]-self.date_range[0]).days)
    
    def __getitem__(self, idx):
       
        data_model = self.ds_model.isel(time = idx)
        data_oi = self.ds_oi.isel(time = idx)

        invar = np.zeros((self.n_channels_in + self.n_channels_oi, self.N_lat, self.N_lon))

        self.ds_masks['longitude'] = data_model['longitude']
        self.ds_masks['latitude'] = data_model['latitude']

        for v, var in enumerate(self.variables_in):
            if var == 'thetao':
                if self.double_sst_mask:
                    invar[v,] = self.ds_obs['sst_double_mask'].isel(time = idx)
                    
                else:
                    invar[v,] = self.ds_obs['sst'].isel(time = idx)
            
            elif var == 'zos':
                invar[v,] = self.ds_obs['ssh'].isel(time = idx)
                
            else:
                invar[v,] = (data_model[var]/self.var_stds[var])
        
        for v, var in enumerate(self.variables_oi):
            invar[v + self.n_channels_in,] = data_oi[var]

        invar = torch.from_numpy(invar.astype(np.float32))
        invar = torch.nan_to_num(invar, nan=0.0)    
        
        return invar