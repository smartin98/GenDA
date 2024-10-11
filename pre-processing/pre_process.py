import os
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter

data_dir = '../input_data/'
wind_dir = ''

ds_g = xr.open_dataset(data_dir + 'cmems_mod_glo_phy_my_0.083deg_P1D-m_so-thetao-zos_70.00W-40.00W_25.00N-45.00N_0.49m_2010-01-01-2020-12-31.nc')
ds15m = xr.open_dataset(data_dir + 'cmems_mod_glo_phy_my_0.083deg_P1D-m_uo-vo_70.00W-40.00W_25.00N-45.00N_13.47-15.81m_2010-01-01-2020-12-31.nc')

# add 15m currents into the surface dataset
ds_g['uo'] = ds15m['uo'].isel(depth = 0, drop = True)
ds_g['vo'] = ds15m['vo'].isel(depth = 0, drop = True)


######### ADD SURFACE WINDS FROM ERA5 (ASSUMING YOU ALREADY HAVE THE DATA LOCALLY)
# download ERA5 uas, vas daily means and store in wind_dir with file name convention wind_u_2019-01.nc, wind_v_2019-01.nc, wind_u_2019-02.nc, ...
files = os.listdir(wind_dir)
files_u = sorted([wind_dir + f for f in files if 'wind_u' in f])
files_v = sorted([wind_dir + f for f in files if 'wind_v' in f])

ds_uas = xr.open_mfdataset(files_u)
ds_vas = xr.open_mfdataset(files_v)

ds_uas = ds_uas.sel(time=slice('2010-01-01', '2020-12-31'), lon = slice(-71,-39), lat = slice(24,46)).rename({'lon':'longitude','lat':'latitude'}).drop(['realization','u10'])
ds_vas = ds_vas.sel(time=slice('2010-01-01', '2020-12-31'), lon = slice(-71,-39), lat = slice(24,46)).rename({'lon':'longitude','lat':'latitude'}).drop(['realization','u10'])

ds_uas = ds_uas.load()
ds_vas = ds_vas.load()

ds_uas = ds_uas.interp_like(ds_g)
ds_vas = ds_vas.interp_like(ds_g)

ds_g['uas'] = ds_uas['uas']
ds_g['vas'] = ds_vas['vas']

###### CALCULATE GEOSTROPHIC CURRENTS

g = 9.81
om = 2*np.pi/86461
f = 2*om*np.sin(np.deg2rad(ds_g['latitude']))

dx = 6378e3 * 2 * np.cos(np.deg2rad(35)) * np.pi / 360
dy = 6378e3 * 2 * np.pi / 360
ds_g['ugos'] = - (g/f) * ds_g['zos'].differentiate('latitude')/dy
ds_g['vgos'] = (g/f) * ds_g['zos'].differentiate('longitude')/dx

ds_g['ugos'] = ds_g['ugos'].transpose('time','latitude','longitude')
ds_g['vgos'] = ds_g['vgos'].transpose('time','latitude','longitude')

ds_g['ugos_smoothed'] = (('time','latitude','longitude'), gaussian_filter(ds_g['ugos'], sigma=[0, 1.5, 1.5]))
ds_g['vgos_smoothed'] = (('time','latitude','longitude'), gaussian_filter(ds_g['vgos'], sigma=[0, 1.5, 1.5]))

ds_g['u_ageo_smoothed_15m'] = ds_g['uo'] - ds_g['ugos_smoothed']
ds_g['v_ageo_smoothed_15m'] = ds_g['vo'] - ds_g['vgos_smoothed']

###### LINEAR REGRESSION FROM WIND STRESS TO CALCULATE EKMAN CURRENT

# estimate surface wind stress:
rho_a = 1.2
c_d = 1.2e-3
ds_g['tau_x'] = rho_a * c_d * np.sqrt(ds_g['uas'] ** 2 + ds_g['vas'] ** 2) * ds_g['uas']
ds_g['tau_y'] = rho_a * c_d * np.sqrt(ds_g['uas'] ** 2 + ds_g['vas'] ** 2) * ds_g['vas']

ue = ds_g['u_ageo_smoothed_15m']
ve = ds_g['v_ageo_smoothed_15m']
tx = ds_g['tau_x']
ty = ds_g['tau_y']

def linear_regression_1d(ue, tx, ty):
    """
    Apply linear regression to 1D arrays ue, tx, and ty across the time dimension
    and return the coefficients a and b.
    """
    # Stack tx and ty to create the feature matrix
    X = np.stack([tx, ty], axis=1)

    if ue[np.isnan(ue)].shape[0]==0:
    
        # Perform linear regression
        reg = LinearRegression().fit(X, ue)
        
        # Return the coefficients a and b
        return reg.coef_[0], reg.coef_[1]
    else:
        return 0, 0

# Use xr.apply_ufunc to apply linear regression over the latitude and longitude dimensions
a_da, b_da = xr.apply_ufunc(
    linear_regression_1d, 
    ue, tx, ty,
    input_core_dims=[['time'], ['time'], ['time']],  # The dimensions over which the function operates
    output_core_dims=[[], []],  # No additional dimensions in the output (a and b are scalars)
    vectorize=True,  # Allow vectorized operation over latitude and longitude
    dask='parallelized',  # Enable parallel computation with dask if needed
    output_dtypes=[float, float],  # The output data types
)

# Assign coordinates and dimensions to the result
a_da = xr.DataArray(a_da, dims=['latitude', 'longitude'], coords={'latitude': ue.latitude, 'longitude': ue.longitude})
b_da = xr.DataArray(b_da, dims=['latitude', 'longitude'], coords={'latitude': ue.latitude, 'longitude': ue.longitude})

# Use xr.apply_ufunc to apply linear regression over the latitude and longitude dimensions
c_da, d_da = xr.apply_ufunc(
    linear_regression_2d, 
    ve, tx, ty,
    input_core_dims=[['time'], ['time'], ['time']],  # The dimensions over which the function operates
    output_core_dims=[[], []],  # No additional dimensions in the output (a and b are scalars)
    vectorize=True,  # Allow vectorized operation over latitude and longitude
    dask='parallelized',  # Enable parallel computation with dask if needed
    output_dtypes=[float, float],  # The output data types
)

# Assign coordinates and dimensions to the result
c_da = xr.DataArray(c_da, dims=['latitude', 'longitude'], coords={'latitude': ve.latitude, 'longitude': ue.longitude})
d_da = xr.DataArray(d_da, dims=['latitude', 'longitude'], coords={'latitude': ve.latitude, 'longitude': ue.longitude})


ds_regress = xr.Dataset({'C_ue_tx': a_da,'C_ue_ty': b_da, 'C_ve_tx': c_da, 'C_ve_ty': d_da})
ds_regress['C_ue_tx'].attrs['Description'] = 'Regression coefficient for eastward ocean current from tau_x: u_ageo_smoothed_15m = C_ue_tx * tau_x + C_ue_ty * tau_y'
ds_regress['C_ue_ty'].attrs['Description'] = 'Regression coefficient for eastward ocean current from tau_y: u_ageo_smoothed_15m = C_ue_tx * tau_x + C_ue_ty * tau_y'
ds_regress['C_ve_tx'].attrs['Description'] = 'Regression coefficient for northward ocean current from tau_x: v_ageo_smoothed_15m = C_ve_tx * tau_x + C_ve_ty * tau_y'
ds_regress['C_ve_ty'].attrs['Description'] = 'Regression coefficient for northward ocean current from tau_y: v_ageo_smoothed_15m = C_ve_tx * tau_x + C_ve_ty * tau_y'
ds_regress.to_netcdf(data_dir + 'glorys_ekman_regression_coefficients.nc')

ds_g['u_ageo_eddy'] = ds_g['u_ageo_smoothed_15m'] - (ds_regress['C_ue_tx'] * ds['tau_x'] + ds_regress['C_ue_ty'] * ds['tau_y'])
ds_g['v_ageo_eddy'] = ds_g['v_ageo_smoothed_15m'] - (ds_regress['C_ve_tx'] * ds['tau_x'] + ds_regress['C_ve_ty'] * ds['tau_y'])

ds_g.to_netcdf(data_dir + 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_70.00W-40.00W_25.00N-45.00N_0.49m_2010-01-01-2020-12-31_wERA5_winds_and_geostrophy_15m_ekman_regression.nc')

ds_m = ds_g.mean(dim = 'time')
ds_m.to_netcdf(data_dir + 'glorys_gulfstream_means_wERA5_winds_and_geostrophy_15m_ekman_regression.nc')

ds_clim = ds_g.groupby('time.month').mean('time')
ds_clim.to_netcdf(data_dir + 'glorys_gulfstream_climatology.nc')