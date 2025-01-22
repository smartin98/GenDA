import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

def gradient_magnitude(x, grid_res = 1/12, lat0 = 38):
    R = 6378e3
    dx = grid_res * R * 2 * np.cos(np.deg2rad(lat0)) * np.pi / 360
    dy = grid_res * R * 2 * np.pi / 360
    return np.sqrt(np.gradient(x, dy)[-2] ** 2 + np.gradient(x, dx)[-1] ** 2)

def vorticity_uv(u,v, grid_res = 1/12, lat0 = 38):
    R = 6378e3
    dx = grid_res * R * 2 * np.cos(np.deg2rad(lat0)) * np.pi / 360
    dy = grid_res * R * 2 * np.pi / 360
    return np.gradient(v, dx)[-1] - np.gradient(u, dy)[-2]

def vorticity_ssh(x, grid_res = 1/12, lat0 = 38):
    R = 6378e3
    dx = grid_res * R * 2 * np.cos(np.deg2rad(lat0)) * np.pi / 360
    dy = grid_res * R * 2 * np.pi / 360
    return np.gradient(np.gradient(x, dy)[-2], dy)[-2] + np.gradient(np.gradient(x, dx)[-1], dx)[-1]

def vorticity_ssh_smoothing(x, grid_res = 1/12, lat0 = 38):
    R = 6378e3
    dx = grid_res * R * 2 * np.cos(np.deg2rad(lat0)) * np.pi / 360
    dy = grid_res * R * 2 * np.pi / 360
    x = gaussian_filter(x, sigma = 1)
    return np.gradient(np.gradient(x, dy)[-2], dy)[-2] + np.gradient(np.gradient(x, dx)[-1], dx)[-1]

def rescale_variable(x, mean, std):
    return x * std + mean

def calculate_R2(pred, truth, axis = None):
    if axis == None:
        return 1 - np.sum((pred - truth)**2)/np.sum(truth**2)
    else:
        return 1 - np.sum((pred - truth)**2, axis = axis)/np.sum(truth**2, axis = axis)

def kld(p, q, x):
    """Return the Kullback-Leibler divergence between 2 PDFs.

    Arguments:
    p -- the first pdf (numpy array of dimension N)
    q -- the second pdf (numpy array of dimension N)
    x -- the x points at which both pdfs are defined (numpy array of dimension N)
    """
    return np.trapz(p * np.log(p/q), x)

def jsd(p, q, x):
    """Return the Jensen-Shannon divergence between 2 PDFs.

    Arguments:
    p -- the first pdf (numpy array of dimension N)
    q -- the second pdf (numpy array of dimension N)
    x -- the x points at which both pdfs are defined (numpy array of dimension N)
    """
    m = 0.5 * (p + q)
    return 0.5 * kld(p, m, x) + 0.5 * kld(q, m, x)

def calculate_jsd(pred, truth, N_samples = 10000, seed = 0, pdf_range = (-7,7), N_pdf_grid = 10000):
    """Estimate the Jensen-Shannon divergence between predictions and truth and also return the pdfs. 

    Arguments:
    p -- the first pdf (numpy array of dimension N)
    q -- the second pdf (numpy array of dimension N)
    x -- the x points at which both pdfs are defined (numpy array of dimension N)
    """
    rng = np.random.default_rng(seed=0)

    nan_mask = (np.isnan(truth)) | (np.isnan(pred))
    
    truth = truth[~nan_mask]
    pred = pred[~nan_mask]

    indices = rng.choice(np.size(truth), N_samples, replace=False)
    truth = truth[indices]
    pred = pred[indices]

    truth_kde = gaussian_kde(truth)
    pred_kde = gaussian_kde(pred)

    x_grid = np.linspace(pdf_range[0], pdf_range[1], num = N_pdf_grid)
    truth_pdf = truth_kde(x_grid)
    pred_pdf = pred_kde(x_grid)

    return jsd(pred_pdf, truth_pdf, x_grid), x_grid, truth_pdf, pred_pdf

def pdf(data, pdf_range = (-7,7), N_pdf_grid = 10000, N_samples = 1000000):
    
    rng = np.random.default_rng(seed=0)

    indices = rng.choice(np.size(data), N_samples, replace=True)
    
    kde = gaussian_kde(data[indices])

    x_grid = np.linspace(pdf_range[0], pdf_range[1], num = N_pdf_grid)

    return kde(x_grid)

def calculate_w(u, v, grid_res = 1/12, lat0 = 38, H = 30):
    """Estimate vertical velocity at depth H given u and v at depth H/2 using 3D incompressibility and w(z=0) = 0.

    Arguments:
    u -- zonal surface current on regular lat-lon grid (numpy array of dimension >=2, last two dimensions are lat, lon)
    v -- meridional surface current on regular lat-lon grid (numpy array of dimension >=2, last two dimensions are lat, lon)
    grid_res -- the resolution of the regular lat-lon grid (float)
    lat0 -- mean latitude of patch (float)
    H -- depth at which to estimate w, should be double the depth at which u/v defined (float)
    """
    R = 6378e3
    dx = grid_res * R * 2 * np.cos(np.deg2rad(lat0)) * np.pi / 360
    dy = grid_res * R * 2 * np.pi / 360

    du_dx = np.gradient(u, dx)[-1]
    dv_dy = np.gradient(v, dy)[-2]

    return H * (du_dx + dv_dy)

def KE_spectrum(u, v, dx, dy):
    fft_u = np.fft.fft2(u)
    fft_u_shifted = np.abs(np.fft.fftshift(fft_u))
    
    fft_v = np.fft.fft2(v)
    fft_v_shifted = np.abs(np.fft.fftshift(fft_v))
    
    fft_magnitude_shifted = 0.5*(fft_u_shifted**2+fft_v_shifted**2)
    
    Nx, Ny = u.shape[-2:]
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    dk = np.abs(kx[1]-kx[0])
    
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    k = np.fft.fftshift(np.sqrt(kx_grid**2 + ky_grid**2))

    return kx_grid, ky_grid, k, fft_magnitude_shifted

def scalar_spectrum(psi, dx, dy):
    fft = np.fft.fft2(psi)
    fft_shifted = np.abs(np.fft.fftshift(fft))
    
    fft_magnitude_shifted = np.abs(fft_shifted)#**2+fft_v_shifted**2)
    
    Nx, Ny = psi.shape[-2:]
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    dk = np.abs(kx[1]-kx[0])
    
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    k = np.fft.fftshift(np.sqrt(kx_grid**2 + ky_grid**2))

    return kx_grid, ky_grid, k, fft_magnitude_shifted

def azimuthal_average(data):
    center = (np.array(data.shape) - 1) / 2.0
    y, x = np.indices(data.shape)
    r_f = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r_f.astype('int')
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def azimuthal_1d_spectrum(k, KE_psd):
    reduce_dims = [i for i in range(len(KE_psd.shape[:-2]))]
    psd = np.mean(KE_psd, axis = tuple(reduce_dims))

    energy_spectrum = azimuthal_average(psd)
    k_spectrum = azimuthal_average(k)
    energy_spectrum = energy_spectrum*2*np.pi*k_spectrum

    return k_spectrum, energy_spectrum

def weighted_resampling(X, W, N_samples=None):
    """Weighted resampling algorithm.
    
    Args:
    X: Dataset.
    W: Weighting vector.
    N_samples: Number of samples to draw (optional).
    
    Returns:
    Resampled dataset.
    """

    if type(X) == list:
        N = np.size(X[0])
    else:
        N = np.size(X)
        
    if N_samples is None:
        N_samples = N
    
    # Normalize weights
    W_normalized = W / np.sum(W)
    
    # Calculate cumulative probabilities
    C = np.cumsum(W_normalized)
    
    # Generate random numbers
    U = np.random.rand(N_samples)
        
    # Sample indices
    sampled_indices = np.searchsorted(C, U, side='right')

    if type(X) == list:

        X_resampled = []
        for x in X:
            # Resampled dataset
            X_resampled.append(x[sampled_indices])
    else:
        X_resampled = X[sampled_indices]
        
    return X_resampled

def calculate_cloudy_uncloudy_R2(preds, truth, sst_mask, N_samples = 1000000):
    cloud_concentration = (1 - np.mean(sst_mask, axis = 0))
    weights_clouded = 1/cloud_concentration
    weights_unclouded = 1/(1 - cloud_concentration)

    weights_cloudy = np.zeros((365, 112, 112))
    weights_uncloudy = np.zeros((365, 112, 112))
    for t in range(365):
        weights_cloudy[t,] = weights_clouded
        weights_uncloudy[t,] = weights_unclouded

    cloudy = weighted_resampling([(preds)[~sst_mask], truth[~sst_mask]], W = weights_cloudy[~sst_mask], N_samples = N_samples)
    uncloudy = weighted_resampling([(preds)[sst_mask], truth[sst_mask]], W = weights_uncloudy[sst_mask], N_samples = N_samples)

    return calculate_R2(cloudy[0], cloudy[1]), calculate_R2(uncloudy[0], uncloudy[1])