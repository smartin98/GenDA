import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

def gradient_magnitude(x):
    return np.sqrt(np.gradient(x)[-2] ** 2 + np.gradient(x)[-1] ** 2)

def vorticity_uv(u,v):
    return np.gradient(v)[-1] - np.gradient(u)[-2]

def vorticity_ssh(x):
    return np.gradient(np.gradient(x)[-2])[-2] + np.gradient(np.gradient(x)[-1])[-1]

def vorticity_ssh_smoothing(x):
    x = gaussian_filter(x, sigma = 1)
    return np.gradient(np.gradient(x)[-2])[-2] + np.gradient(np.gradient(x)[-1])[-1]

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
    
    
    
    