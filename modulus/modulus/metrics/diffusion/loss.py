# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import random
from typing import Callable, Optional, Union

import numpy as np
import torch



def physics_features(data, geography_scalings, var_stds, physics_scalings):
    """Calculates physical features from oceanographic data.

    This function computes various physical features, including advective fluxes,
    rho gradients, vorticity, strain, geostrophic and ageostrophic components
    of these quantities. The features are normalized using provided statistics.

    Args:
        data: Input data tensor of shape (batch_size, num_vars, lat, lon).
        geography_scalings: Dictionary containing grid spacing (dx, dy) and
                           Coriolis parameter (gof).
        var_stds: Dictionary containing standard deviations for variables.
        physics_scalings: Dictionary containing mean and standard deviation
                          for each calculated physical feature.

    Returns:
        Tensor of shape (batch_size, num_features, lat, lon) containing
        normalized physical features.
    """
    
    ssh = var_stds['zos'] * data[:,0,]
    sst = var_stds['thetao'] * data[:,1,]
    sss = var_stds['so']* data[:,2,]
    u = var_stds['uo']* data[:,3,]
    v = var_stds['vo']* data[:,4,]
    dx = geography_scalings['dx']
    dy = geography_scalings['dy']
    gof = geography_scalings['gof']
    
    # SST-U-V connections
    #zonal advective sst flux: u*dSST/dx
    sst_xflux = u*torch.gradient(sst,spacing = dx)[2]
    #meridional advective sst flux: v*dSST/dy
    sst_yflux = v*torch.gradient(sst,spacing = dy)[1]

    # SSS-U-V connections
    #zonal advective sss flux: u*dSSS/dx
    sss_xflux = u*torch.gradient(sss,spacing = dx)[2]
    #meridional advective sss flux: v*dSSS/dy
    sss_yflux = v*torch.gradient(sss,spacing = dy)[1]

    #SSS-SST connections
    #zonal rho gradient: -dSST/dx + dSSS/dx
    drho_dx = -torch.gradient(sst,spacing = dx)[2] + torch.gradient(sss,spacing = dx)[2]
    #meridional rho gradient: -dSST/dy + dSSS/dy
    drho_dy = -torch.gradient(sst,spacing = dy)[1] + torch.gradient(sss,spacing = dy)[1]

    #U-V connections
    #vorticity: dv/dx - du/dy
    vort = torch.gradient(v,spacing = dx)[2] - torch.gradient(u,spacing = dy)[1]
    #normal strain: du/dx - dv/dy
    sn = torch.gradient(u,spacing = dx)[2] - torch.gradient(v,spacing = dy)[1]
    #normal strain: dv/dx + du/dy
    ss = torch.gradient(v,spacing = dx)[2] + torch.gradient(u,spacing = dy)[1]

    #SSH-SST connections
    #geostrophic zonal advective sst flux: -(g/f)*dSSH/dy*dSST/dx
    geo_sst_xflux = -gof*torch.gradient(ssh,spacing = dy)[1]*torch.gradient(sst,spacing = dx)[2]
    #geostrophic meridional advective sst flux: (g/f)*dSSH/dx*dSST/dy
    geo_sst_yflux = gof*torch.gradient(ssh,spacing = dx)[2]*torch.gradient(sst,spacing = dy)[1]

    #SSH-SSS connections
    #geostrophic zonal advective sss flux: -(g/f)*dSSH/dy*dSSS/dx
    geo_sss_xflux = -gof*torch.gradient(ssh,spacing = dy)[1]*torch.gradient(sss,spacing = dx)[2]
    #geostrophic meridional advective sss flux: (g/f)*dSSH/dx*dSSS/dy
    geo_sss_yflux = gof*torch.gradient(ssh,spacing = dx)[2]*torch.gradient(sss,spacing = dy)[1]

    #SSH-U-V connections
    #ageostrophic vorticity: (dv/dx-du/dy) - (g/f)*(d2SSH/dx2 + d2SSH/dy2)
    ageo_vort = vort - gof*(torch.gradient(torch.gradient(ssh,spacing = dx)[2],spacing = dx)[2] + torch.gradient(torch.gradient(ssh,spacing = dy)[1],spacing = dy)[1])
    #ageostrophic normal strain: (du/dx-dv/dy) - (g/f)*( - 2 * d2SSH/dxy)
    ageo_sn = sn + gof*2*torch.gradient(torch.gradient(ssh,spacing = dy)[1],spacing = dx)[2]
    #ageostrophic shear strain: (dv/dx+du/dy) - (g/f)*(d2SSH_dx2 - d2SSH_dy2)
    ageo_ss = ss - gof*(torch.gradient(torch.gradient(ssh, spacing = dx)[2], spacing = dx)[2] - torch.gradient(torch.gradient(ssh, spacing = dy)[1], spacing = dy)[1])
    
    return torch.stack(
        ((sst_xflux-physics_scalings['sst_xflux']['mean'])/physics_scalings['sst_xflux']['std'], 
         (sst_yflux-physics_scalings['sst_yflux']['mean'])/physics_scalings['sst_yflux']['std'], 
         (sss_xflux-physics_scalings['sss_xflux']['mean'])/physics_scalings['sss_xflux']['std'], 
         (sss_yflux-physics_scalings['sss_yflux']['mean'])/physics_scalings['sss_yflux']['std'], 
         (drho_dx-physics_scalings['drho_dx']['mean'])/physics_scalings['drho_dx']['std'], 
         (drho_dy-physics_scalings['drho_dy']['mean'])/physics_scalings['drho_dy']['std'], 
         (vort-physics_scalings['vort']['mean'])/physics_scalings['vort']['std'], 
         (sn-physics_scalings['sn']['mean'])/physics_scalings['sn']['std'], 
         (ss-physics_scalings['ss']['mean'])/physics_scalings['ss']['std'], 
         (geo_sst_xflux-physics_scalings['geo_sst_xflux']['mean'])/physics_scalings['geo_sst_xflux']['std'], 
         (geo_sst_yflux-physics_scalings['geo_sst_yflux']['mean'])/physics_scalings['geo_sst_yflux']['std'], 
         (geo_sss_xflux-physics_scalings['geo_sss_xflux']['mean'])/physics_scalings['geo_sss_xflux']['std'], 
         (geo_sss_yflux-physics_scalings['geo_sss_yflux']['mean'])/physics_scalings['geo_sss_yflux']['std'], 
         (ageo_vort-physics_scalings['ageo_vort']['mean'])/physics_scalings['ageo_vort']['std'], 
         (ageo_sn-physics_scalings['ageo_sn']['mean'])/physics_scalings['ageo_sn']['std'], 
         (ageo_ss-physics_scalings['ageo_ss']['mean'])/physics_scalings['ageo_ss']['std'])
        , dim = 1)

def dummy_features(data):
    """Calculates physical features from oceanographic data.

    This function computes various physical features, including advective fluxes,
    rho gradients, vorticity, strain, geostrophic and ageostrophic components
    of these quantities. The features are normalized using provided statistics.

    Args:
        data: Input data tensor of shape (batch_size, num_vars, lat, lon).
        geography_scalings: Dictionary containing grid spacing (dx, dy) and
                           Coriolis parameter (gof).
        var_stds: Dictionary containing standard deviations for variables.
        physics_scalings: Dictionary containing mean and standard deviation
                          for each calculated physical feature.

    Returns:
        Tensor of shape (batch_size, num_features, lat, lon) containing
        normalized physical features.
    """
    
    ssh = data[:,0,]
    sst = data[:,1,]
    sss = data[:,2,]
    u = data[:,3,]
    v = data[:,4,]
    
    # SST-U-V connections
    sst_u = u + sst
    sst_v = v + sst
    sst_u_m = u - sst
    sst_v_m = v - sst

    # SSS-U-V connections
    sss_u = u + sss
    sss_v = v + sss
    sss_u_m = u - sss
    sss_v_m = v - sss


    #SSS-SST connections
    sst_sss = sst + sss
    sst_sss_m = sst - sss

    #U-V connections
    u_v = u + v
    u_v_m = u - v

    #SSH-SST connections
    sst_ssh = sst + ssh
    sst_ssh_m = sst - ssh

    #SSH-SSS connections
    sss_ssh = sss + ssh
    sss_ssh_m = sss - ssh

    #SSH-U-V connections
    ssh_u = ssh + u
    ssh_v = ssh + v
    ssh_u_m = ssh - u
    ssh_v_m = ssh - v

    return torch.stack((sst_u, 
                        sst_v, 
                        sss_u, 
                        sss_v, 
                        sst_sss, 
                        u_v, 
                        sst_ssh, 
                        sss_ssh, 
                        ssh_u, 
                        ssh_v,
                        sst_u_m, 
                        sst_v_m, 
                        sss_u_m, 
                        sss_v_m, 
                        sst_sss_m, 
                        u_v_m, 
                        sst_ssh_m, 
                        sss_ssh_m, 
                        ssh_u_m, 
                        ssh_v_m,
                       ), dim = 1)


class EDMLossPhysics:
    """
    Loss function proposed in the EDM paper.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(
        self, var_stds: dict = {}, geography_scalings: dict = {}, physics_scalings: dict = {}, lambda_physics: float = 1, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5
    ):
        self.var_stds = var_stds
        self.geography_scalings = geography_scalings 
        self.physics_scalings = physics_scalings
        self.lambda_physics = lambda_physics
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        

    def __call__(self, net, images, labels=None, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the EDM formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'P_mean' and 'P_std' random values. The calculated loss is weighted as a
        function of 'sigma' and 'sigma_data'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma[:,0,0,0], labels, augment_labels=augment_labels) # fixed Positional embedding shape error in modulus release

        # extend state vector by adding physics-informed features that link the different channels together:
        physics_y = physics_features(y, geography_scalings = self.geography_scalings, var_stds = self.var_stds, physics_scalings = self.physics_scalings)
        physics_D_yn = physics_features(D_yn, geography_scalings = self.geography_scalings, var_stds = self.var_stds, physics_scalings = self.physics_scalings)
         
        loss = weight * torch.cat((((1/(1+self.lambda_physics))*y.shape[1]/(y.shape[1] + physics_y.shape[1]))*(D_yn - y) ** 2, 
                                   (self.lambda_physics/(1+self.lambda_physics))*(physics_y.shape[1]/(y.shape[1] + physics_y.shape[1]))*(physics_D_yn - physics_y)**2), 
                                  dim = 1)

        return loss




class VPLoss:
    """
    Loss function corresponding to the variance preserving (VP) formulation.

    Parameters
    ----------
    beta_d: float, optional
        Coefficient for the diffusion process, by default 19.9.
    beta_min: float, optional
        Minimum bound, by defaults 0.1.
    epsilon_t: float, optional
        Small positive value, by default 1e-5.

    Note:
    -----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.

    """

    def __init__(
        self, beta_d: float = 19.9, beta_min: float = 0.1, epsilon_t: float = 1e-5
    ):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(
        self,
        net: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        augment_pipe: Optional[Callable] = None,
    ):
        """
        Calculate and return the loss corresponding to the variance preserving (VP)
        formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'epsilon_t' and random values. The calculated loss is weighted based on the
        inverse of 'sigma^2'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(
        self, t: Union[float, torch.Tensor]
    ):  # NOTE: also exists in preconditioning
        """
        Compute the sigma(t) value for a given t based on the VP formulation.

        The function calculates the noise level schedule for the diffusion process based
        on the given parameters `beta_d` and `beta_min`.

        Parameters
        ----------
        t : Union[float, torch.Tensor]
            The timestep or set of timesteps for which to compute sigma(t).

        Returns
        -------
        torch.Tensor
            The computed sigma(t) value(s).
        """
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


class VELoss:
    """
    Loss function corresponding to the variance exploding (VE) formulation.

    Parameters
    ----------
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.

    Note:
    -----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    """

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 100.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the variance exploding (VE)
        formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'sigma_min' and 'sigma_max' and random values. The calculated loss is weighted
        based on the inverse of 'sigma^2'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLoss:
    """
    Loss function proposed in the EDM paper.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(
        self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the EDM formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'P_mean' and 'P_std' random values. The calculated loss is weighted as a
        function of 'sigma' and 'sigma_data'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma[:,0,0,0], labels, augment_labels=augment_labels) # fixed Positional embedding shape error in modulus release
        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLossSR:
    """
    Variation of the loss function proposed in the EDM paper for Super-Resolution.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the EDM formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'P_mean' and 'P_std' random values. The calculated loss is weighted as a
        function of 'sigma' and 'sigma_data'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, y_lr, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class RegressionLoss:
    """
    Regression loss function for the U-Net for deterministic predictions.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        """
        Calculate and return the loss for the U-Net for deterministic predictions.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        img_clean: torch.Tensor
            Input images (high resolution) to the neural network.

        img_lr: torch.Tensor
            Input images (low resolution) to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (
            1.0  # (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        )

        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        input = torch.zeros_like(y, device=img_clean.device)
        D_yn = net(input, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)

        return loss


class ResLoss:
    """
    Mixture loss function for denoising score matching.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        regression_net,
        img_shape_x,
        img_shape_y,
        patch_shape_x,
        patch_shape_y,
        patch_num,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
    ):
        self.unet = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.patch_shape_x = patch_shape_x
        self.patch_shape_y = patch_shape_y
        self.patch_num = patch_num
        self.hr_mean_conditioning = hr_mean_conditioning

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        """
        Calculate and return the loss for denoising score matching.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        img_clean: torch.Tensor
            Input images (high resolution) to the neural network.

        img_lr: torch.Tensor
            Input images (low resolution) to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """

        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr

        # global index
        b = y.shape[0]
        Nx = torch.arange(self.img_shape_x).int()
        Ny = torch.arange(self.img_shape_y).int()
        grid = torch.stack(torch.meshgrid(Ny, Nx, indexing="ij"), dim=0)[
            None,
        ].expand(b, -1, -1, -1)

        # form residual
        y_mean = self.unet(
            torch.zeros_like(y, device=img_clean.device),
            y_lr_res,
            sigma,
            labels,
            augment_labels=augment_labels,
        )

        y = y - y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((y_mean, y_lr), dim=1).contiguous()
        global_index = None
        # patchified training
        # conditioning: cat(y_mean, y_lr, input_interp, pos_embd), 4+12+100+4
        if (
            self.img_shape_x != self.patch_shape_x
            or self.img_shape_y != self.patch_shape_y
        ):
            c_in = y_lr.shape[1]
            c_out = y.shape[1]
            rnd_normal = torch.randn(
                [img_clean.shape[0] * self.patch_num, 1, 1, 1], device=img_clean.device
            )
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            weight = (sigma**2 + self.sigma_data**2) / (
                sigma * self.sigma_data
            ) ** 2

            # global interpolation
            input_interp = torch.nn.functional.interpolate(
                img_lr,
                (self.patch_shape_y, self.patch_shape_x),
                mode="bilinear",
            )

            # patch generation from a single sample (not from random samples due to memory consumption of regression)
            y_new = torch.zeros(
                b * self.patch_num,
                c_out,
                self.patch_shape_y,
                self.patch_shape_x,
                device=img_clean.device,
            )
            y_lr_new = torch.zeros(
                b * self.patch_num,
                c_in + input_interp.shape[1],
                self.patch_shape_y,
                self.patch_shape_x,
                device=img_clean.device,
            )
            global_index = torch.zeros(
                b * self.patch_num,
                2,
                self.patch_shape_y,
                self.patch_shape_x,
                dtype=torch.int,
                device=img_clean.device,
            )
            for i in range(self.patch_num):
                rnd_x = random.randint(0, self.img_shape_x - self.patch_shape_x)
                rnd_y = random.randint(0, self.img_shape_y - self.patch_shape_y)
                y_new[b * i : b * (i + 1),] = y[
                    :,
                    :,
                    rnd_y : rnd_y + self.patch_shape_y,
                    rnd_x : rnd_x + self.patch_shape_x,
                ]
                global_index[b * i : b * (i + 1),] = grid[
                    :,
                    :,
                    rnd_y : rnd_y + self.patch_shape_y,
                    rnd_x : rnd_x + self.patch_shape_x,
                ]
                y_lr_new[b * i : b * (i + 1),] = torch.cat(
                    (
                        y_lr[
                            :,
                            :,
                            rnd_y : rnd_y + self.patch_shape_y,
                            rnd_x : rnd_x + self.patch_shape_x,
                        ],
                        input_interp,
                    ),
                    1,
                )
            y = y_new
            y_lr = y_lr_new
        latent = y + torch.randn_like(y) * sigma
        D_yn = net(
            latent,
            y_lr,
            sigma,
            labels,
            global_index=global_index,
            augment_labels=augment_labels,
        )
        loss = weight * ((D_yn - y) ** 2)

        return loss


class VELoss_dfsr:
    """
    Loss function for dfsr model, modified from class VELoss.

    Parameters
    ----------
    beta_start : float
        Noise level at the initial step of the forward diffusion process, by default 0.0001.
    beta_end : float
        Noise level at the Final step of the forward diffusion process, by default 0.02.
    num_diffusion_timesteps : int
        Total number of forward/backward diffusion steps, by default 1000.


    Note:
    -----
    Reference: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models.
    Advances in neural information processing systems. 2020;33:6840-51.
    """

    def __init__(
        self,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_diffusion_timesteps: int = 1000,
    ):
        # scheduler for diffusion:
        self.beta_schedule = "linear"
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_diffusion_timesteps = num_diffusion_timesteps
        betas = self.get_beta_schedule(
            beta_schedule=self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            num_diffusion_timesteps=self.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = betas.shape[0]

    def get_beta_schedule(
        self, beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps
    ):
        """
        Compute the variance scheduling parameters {beta(0), ..., beta(t), ..., beta(T)}
        based on the VP formulation.

        beta_schedule: str
            Method to construct the sequence of beta(t)'s.
        beta_start: float
            Noise level at the initial step of the forward diffusion process, e.g., beta(0)
        beta_end: float
            Noise level at the final step of the forward diffusion process, e.g., beta(T)
        num_diffusion_timesteps: int
            Total number of forward/backward diffusion steps
        """

        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if beta_schedule == "quad":
            betas = (
                np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        elif beta_schedule == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(
                num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        if betas.shape != (num_diffusion_timesteps,):
            raise ValueError(
                f"Expected betas to have shape ({num_diffusion_timesteps},), "
                f"but got {betas.shape}"
            )
        return betas

    def __call__(self, net, images, labels, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the variance preserving
        formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the noise samples added
        to the t-th step of the diffusion process.
        The noise level is determined by 'beta_t' based on the given parameters 'beta_start',
        'beta_end' and the current diffusion timestep t.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input fluid flow data samples to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input fluid flow data samples. Not required for dfsr.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(images.size(0) // 2 + 1,)
        ).to(images.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[: images.size(0)]
        e = torch.randn_like(images)
        b = self.betas.to(images.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = images * a.sqrt() + e * (1.0 - a).sqrt()

        output = net(x, t, labels)
        loss = (e - output).square()

        return loss
