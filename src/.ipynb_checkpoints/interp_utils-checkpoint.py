import numpy as np
from numpy.random import randint
import pyproj
import scipy.spatial.transform 
import scipy.stats as stats
from scipy import interpolate
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import xarray as xr 
from datetime import date, timedelta
import os

# function to list all files within a directory including within any subdirectories
def GetListOfFiles(dirName, ext = '.nc'):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetListOfFiles(fullPath)
        else:
            if fullPath.endswith(ext):
                allFiles.append(fullPath)               
    return allFiles

# Define the pyproj transformer objects used to transform coordinates between (lat,long,alt) and ECEF in both directions
transformer_ll2xyz = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
transformer_xyz2ll = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        )

# convert ECEF coords to lat,lon:
def xyz2ll(x,y,z, lat_org, lon_org, alt_org, transformer1, transformer2):

    # transform origin of local tangent plane to ECEF coordinates (https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
    x_org, y_org, z_org = transformer1.transform( lon_org,lat_org,  alt_org,radians=False)
    ecef_org=np.array([[x_org,y_org,z_org]]).T

    # define 3D rotation required to transform between ECEF and ENU coordinates (https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
    rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()
    rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()
    rotMatrix = rot1.dot(rot3)

    # transform ENU coords to ECEF by rotating
    ecefDelta = rotMatrix.T.dot(np.stack([x,y,np.zeros_like(x)],axis=-1).T)
    # add offset of all corrds on tangent plane to get all points in ECEF
    ecef = ecefDelta+ecef_org
    # transform to geodetic coordinates
    lon, lat, alt = transformer2.transform( ecef[0,:],ecef[1,:],ecef[2,:],radians=False)
    # only return lat, lon since we're interested in points on Earth. 
    # N.B. this amounts to doing an inverse stereographic projection from ENU to lat, lon so shouldn't be used to directly back calculate lat, lon from tangent plane coords
    # this is instead achieved by binning the data's lat/long variables onto the grid in the same way as is done for the variable of interest
    return lat, lon

def ll2xyz(lat, lon, alt, lat_org, lon_org, alt_org, transformer):

    # transform geodetic coords to ECEF (https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
    x, y, z = transformer.transform( lon,lat, np.zeros_like(lon),radians=False)
    x_org, y_org, z_org = transformer.transform( lon_org,lat_org,  alt_org,radians=False)
    # define position of all points relative to origin of local tangent plane
    vec=np.array([[ x-x_org, y-y_org, z-z_org]]).T

    # define 3D rotation required to transform between ECEF and ENU coordinates (https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
    rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()
    rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()
    rotMatrix = rot1.dot(rot3)    

    # rotate ECEF coordinates to ENU
    enu = rotMatrix.dot(vec).T
    X, Y, Z = enu[0, :, 0], enu[0, :, 1], enu[0, :, 2]
    
    return X, Y, Z

# generate coords for points on a square with prescribed side length and center
def box(x_bounds, y_bounds, refinement=100):
    xs = np.zeros(int(4*refinement))
    ys = np.zeros(int(4*refinement))
    
    xs[:refinement] = np.linspace(x_bounds[0], x_bounds[-1], num=refinement)
    ys[:refinement] = np.linspace(y_bounds[0], y_bounds[0], num=refinement)
                
    xs[refinement:2*refinement] = np.linspace(x_bounds[-1], x_bounds[-1], num=refinement)
    ys[refinement:2*refinement] = np.linspace(y_bounds[0], y_bounds[-1], num=refinement)
                
    xs[2*refinement:3*refinement] = np.linspace(x_bounds[-1], x_bounds[0], num=refinement)
    ys[2*refinement:3*refinement] = np.linspace(y_bounds[-1], y_bounds[-1], num=refinement)
    
    xs[3*refinement:] = np.linspace(x_bounds[0], x_bounds[0], num=refinement)
    ys[3*refinement:] = np.linspace(y_bounds[-1], y_bounds[0], num=refinement)
    
    return xs, ys

def grid_sst_hr(ds, n_t, n, L_x, L_y, lon0, lat0, coord_grid):
    
    lon_grid = coord_grid[:,:,0].ravel()
    lat_grid = coord_grid[:,:,1].ravel()
    lat_max = np.max(lat_grid)+0.1
    lat_min = np.min(lat_grid)-0.1

    
    ds = ds.sel(lat=slice(lat_min,lat_max), drop = True)
    
    if ((np.size(lon_grid[lon_grid>175])>0) and (np.size(lon_grid[lon_grid<-175])>0)):
        long_max_unshifted = np.max(lon_grid[lon_grid<0]) + 0.1
        long_min_unshifted = np.min(lon_grid[lon_grid>0]) - 0.1
        
    else:
        long_max_unshifted = np.max(lon_grid) + 0.1
        long_min_unshifted = np.min(lon_grid) - 0.1
 
    if long_max_unshifted>long_min_unshifted:
        ds = ds.isel(lon = (ds.lon < long_max_unshifted) & (ds.lon > long_min_unshifted),drop = True)
    else:
        ds1 = ds.isel(lon = (ds.lon < long_max_unshifted),drop=True)
        ds2 = ds.isel(lon = (ds.lon > long_min_unshifted),drop=True)
        ds = xr.concat([ds1,ds2],'lon') 
    
    ds = ds.load()

    ds['lon'] = (ds['lon']-lon0+180)%360-180

    lon = ds['lon']
    lat = ds['lat']
    lon, lat = np.meshgrid(lon, lat)

    lon = lon.flatten()
    lat = lat.flatten()
    
    sst_list = []
    for t in range(n_t):
        sst = np.array(ds['analysed_sst'].isel(time=t)).ravel()
        sst[np.isnan(sst)] = 0
        sst_list.append(sst)

    # calculate ENU coords of data on tangent plane
    x,y,_ = ll2xyz(lat, lon, 0, lat0, 0, 0, transformer_ll2xyz)
    sst_grids, _,_,_ = stats.binned_statistic_2d(x, y, sst_list, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
    
    for i,sst_grid in enumerate(sst_grids):
        sst_grid = np.rot90(sst_grid)
        sst_grid[sst_grid<270] = 0
        sst_grids[i] = sst_grid
    
    return sst_grids

def load_multisat_ssh(ssh_files):
    datasets = []
    ds = xr.open_dataset(ssh_files[0])
    sat = str(os.path.dirname(ssh_files[0]).split('cmems_sla/')[1])
    ds['time_variable'] = ds['time']
    ds['time'] = np.arange(ds['time'].shape[0])
    ds['sat'] = xr.DataArray([sat] * ds['time'].shape[0], dims=['time'])    
    ds = ds.rename({'time':'n_obs'})
    ds = ds.rename({'time_variable':'time'})
    datasets.append(ds)
    length = ds['n_obs'].shape[0]
    for f in ssh_files[1:]:
        ds = xr.open_dataset(f)
        sat = str(os.path.dirname(f).split('cmems_sla/')[1])
        ds['time_variable'] = ds['time']
        ds['time'] = np.arange(length,length+ds['time'].shape[0])
        ds['sat'] = xr.DataArray([sat] * ds['time'].shape[0], dims=['time'])
        ds = ds.rename({'time':'n_obs'})
        ds = ds.rename({'time_variable':'time'})
        datasets.append(ds)
        length+=ds['n_obs'].shape[0]

    ds = xr.concat(datasets,dim='n_obs')
    ds = ds.sortby('time')
    ds['n_obs'] = ds['time']
    ds = ds.drop_vars(['time'])
    ds = ds.rename({'n_obs':'time'})
    
    return ds

def extract_tracked(ds, date_min, date_max, L_x, L_y, lon0, lat0, transformer_ll2xyz, coord_grid, filtered = False, withhold_sats=None):
    ds = ds.sel(time=slice(date_min, date_max))
    # shift to Greenwich to avoid dateline issues
    ds['longitude'] = (ds['longitude'] % 360 - lon0 + 180) % 360 - 180
    
    lon_grid = coord_grid[:,:,0].ravel()
    lat_grid = coord_grid[:,:,1].ravel()
    lat_max = np.max(lat_grid)+1
    lat_min = np.min(lat_grid)-1
    
    lon_grid_shifted = (lon_grid % 360 - lon0 + 180) % 360 - 180
    lon_max = np.max(lon_grid_shifted)+1
    lon_min = np.min(lon_grid_shifted)-1
    
    lon_mask = (ds['longitude'] >= lon_min) & (ds['longitude'] <= lon_max)
    lat_mask = (ds['latitude'] >= lat_min) & (ds['latitude'] <= lat_max)
    mask = lon_mask & lat_mask
    
    ds = ds.isel(time=mask)
    
    longitude = np.array(ds['longitude']).ravel() 
    latitude = np.array(ds['latitude']).ravel()
    if filtered:
        sla = np.array(ds['sla_filtered']).ravel()
    else:
        sla = np.array(ds['sla_unfiltered']).ravel()
    
    sat = np.array(ds['sat']).ravel()
    t = np.array(ds['time']).ravel()

    x, y, z = ll2xyz(latitude, longitude, 0, lat0, 0, 0, transformer_ll2xyz)
    mask = (z > -1000e3) & (x < L_x / 2) & (x > -L_x / 2) & (y < L_y / 2) & (y > -L_y / 2)
    x, y, sla, sat, t = x[mask], y[mask], sla[mask], sat[mask], t[mask]

    if withhold_sats is not None:
        mask = np.isin(sat, withhold_sats, invert=True)  
        x_in, y_in, t_in, sla_in = x[mask], y[mask], t[mask], sla[mask]
        x_out, y_out, t_out, sla_out = x[~mask], y[~mask], t[~mask], sla[~mask]
        tracks_in = {'x': x_in, 'y': y_in, 'time': t_in, 'sla': sla_in}
        tracks_out = {'x': x_out, 'y': y_out, 'time': t_out, 'sla': sla_out}
        return tracks_in, tracks_out

    else:
        tracks = {'x': x, 'y': y, 'time': t, 'sla': sla}
        return tracks


def grid_ssh(tracks, n, N_t, L_x, L_y, start_date):
    x = tracks['x']
    y = tracks['y']
    ssh = tracks['sla']

    days_since_start = (tracks['time'] - np.datetime64(start_date)) / np.timedelta64(1, 'D')
    days_since_start = days_since_start.astype(int)
    
    # group by day
    groups = [
        (x[days_since_start == day], y[days_since_start == day], ssh[days_since_start == day]) 
        for day in range(N_t)
    ]

    # mask empty days
    empty_mask = [len(group[0]) == 0 for group in groups]

    data_final = np.zeros((N_t, n, n))
    
    valid_groups = [
        group if not empty else (np.zeros(1), np.zeros(1), np.zeros(1)) 
        for group, empty in zip(groups, empty_mask)
    ]

    input_grids = [
        np.rot90(stats.binned_statistic_2d(x, y, ssh, statistic='mean', bins=n, range=[[-L_x/2, L_x/2], [-L_y/2, L_y/2]])[0])
        for x, y, ssh in valid_groups
    ]


    data_final[empty_mask] = 0

    if np.sum(~np.array(empty_mask))>0:
        data_final[~np.array(empty_mask)] = np.stack(input_grids, axis=0)
    else:
        data_final = np.zeros((N_t, n, n))
        
    data_final[np.isnan(data_final)] = 0

    return data_final

def normalise_ssh(ssh, mean_ssh, std_ssh):    
    return (ssh-mean_ssh)/std_ssh

def rescale_x(x, L_x, n):
    return (x + 0.5*L_x)*(n - 1)/L_x

def rescale_y(y, L_y, n): 
    return (-y + 0.5*L_y)*(n - 1)/L_y

# def reformat_output_tracks(tracks, max_outvar_length, N_t, n, L_x, L_y, start_date, mean_ssh, std_ssh, filtered=False):
#     x = tracks['x']
#     y = tracks['y']
#     t = tracks['time']
#     ssh = tracks['sla_filtered'] if filtered else tracks['sla_unfiltered']
#     days_since_start = [dt - np.datetime64(start_date) for dt in tracks['time']]
#     days_since_start = (np.array(days_since_start).astype('timedelta64[D]') / np.timedelta64(1, 'D')).astype('int').tolist()
#     unique_values = np.unique(days_since_start)
#     missing_days = [t for t in range(N_t) if t not in unique_values]
#     first_indices = [np.where(days_since_start == value)[0][0] for value in unique_values]
#     data_final = np.zeros((N_t,max_outvar_length,3))
#     missed_days = 0
#     for day in range(N_t):
#         if day not in missing_days:
#             if day == N_t-1:
#                 ssh_loop = ssh[first_indices[day-missed_days]:]
#                 x_loop = x[first_indices[day-missed_days]:]
#                 y_loop = y[first_indices[day-missed_days]:]
#             else:
#                 ssh_loop = ssh[first_indices[day-missed_days]:first_indices[day-missed_days+1]]
#                 x_loop = x[first_indices[day-missed_days]:first_indices[day-missed_days+1]]
#                 y_loop = y[first_indices[day-missed_days]:first_indices[day-missed_days+1]]

#             x_loop = x_loop[~np.isnan(ssh_loop)]
#             y_loop = y_loop[~np.isnan(ssh_loop)]
#             ssh_loop = ssh_loop[~np.isnan(ssh_loop)]
            
#             x_loop = rescale_x(x_loop, L_x, n)
#             y_loop = rescale_y(y_loop, L_y, n)
#             ssh_loop = normalise_ssh(ssh_loop, mean_ssh, std_ssh)
            
#             n_obs = ssh_loop.shape[0]
#             if n_obs<max_outvar_length:
#                 data_final[day-missed_days,:n_obs,0] = x_loop
#                 data_final[day-missed_days,:n_obs,1] = y_loop
#                 data_final[day-missed_days,:n_obs,2] = ssh_loop
#             else:
#                 data_final[day-missed_days,:,0] = x_loop[:max_outvar_length]
#                 data_final[day-missed_days,:,1] = y_loop[:max_outvar_length]
#                 data_final[day-missed_days,:,2] = ssh_loop[:max_outvar_length]
            
                
#         else:
#             missed_days+=1
            
#     return data_final

def reformat_output_tracks(tracks, max_outvar_length, N_t, n, L_x, L_y, start_date, mean_ssh, std_ssh):
    x = tracks['x']
    y = tracks['y']
    ssh = tracks['sla']

    days_since_start = (tracks['time'] - np.datetime64(start_date)) / np.timedelta64(1, 'D')
    days_since_start = days_since_start.astype(int)

    data_final = np.zeros((N_t, max_outvar_length, 3))
    
    for day in range(N_t):
        mask = days_since_start == day
        x_loop = x[mask]
        y_loop = y[mask]
        ssh_loop = ssh[mask]
        
        valid_mask = ~np.isnan(ssh_loop)
        x_loop = x_loop[valid_mask]
        y_loop = y_loop[valid_mask]
        ssh_loop = ssh_loop[valid_mask]

        if x_loop.size > 0:  
            x_loop = rescale_x(x_loop, L_x, n)
            y_loop = rescale_y(y_loop, L_y, n)
            ssh_loop = normalise_ssh(ssh_loop, mean_ssh, std_ssh)

            n_obs = min(x_loop.shape[0], max_outvar_length)
            data_final[day, :n_obs, 0] = x_loop[:n_obs]
            data_final[day, :n_obs, 1] = y_loop[:n_obs]
            data_final[day, :n_obs, 2] = ssh_loop[:n_obs]

    return data_final