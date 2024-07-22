# GenDA

GenDA - Generative Data Assimilation. 

Experiments in generative neural data assimilation for multi-modal surface ocean state estimation.

Given high-resolution training data from eddy-resolving numerical simulations, train a generative model to produce realistic multi-modal surface snapshots from the model (e.g. sea surface height, temperature, salinity, & surface currents). Can we then use this generative model to estimate poorly-observed quantities (e.g. surface currents/salinity) from satellite observables (e.g. sea surface height and temperature)?

## Idea #1: Variational Autoencoder + Latent Space Inversion
*Step 1:* Train VAE on simulation snapshots to learn well-structured latent space and a decoder that maps latent vector to realistic multi-modal snapshots.
*Step 2:* Use latent space inversion (gradient descent with decoder network weights fixed and latent vector components learned) to find optimal latent vector given partial observations.

*Preliminary findings:* VAE can produce realistic multi-modal samples, latent space inversion then allows large-scale currents to be inferred from SSH/SST but latent space convergence not well behaved.

## Idea #2: Diffusion Model (e.g. https://arxiv.org/abs/2406.16947)

*Step 1:* Train unconditional diffusion model to produce realistic multi-modal samples.
*Step 2:* Nudge trained model to generate samples that best match partial observations by optimizing over the input noise (e.g. https://arxiv.org/abs/2406.16947).

*Work In Progress.*