# GenDA

GenDA - Generative Data Assimilation. 

:warning: *Work In Progress.* :warning:

![Alt text](images/GenDA_generation_OSSE.gif)

Experiments in generative neural data assimilation for multi-modal surface ocean state estimation. These experiments will be more thoroughly described in a pre-print which is in prep. along with a full code release.

**The problem:** Estimate the multi-modal dynamical state of the surface ocean (sea surface height, temperature, salinity, and surface currents) from sparse satellite observations of sea surface height and temperature and low-resolution objective analysis products for sea surface height, temperature, and salinity.

![Alt text](images/osse_inputs_outputs.png)

**The approach:** Given high-resolution training data from eddy-resolving numerical simulations, train a generative model to produce realistic multi-modal surface snapshots from the model (e.g. sea surface height, temperature, salinity, & surface currents). Can we then use this generative model to estimate poorly-observed quantities (e.g. surface currents/salinity) from satellite observables (e.g. sea surface height and temperature)? 

Motivations for a generative approach vs regression approach:
1. Predicting single value with regression approach smooths out small-scale features, impacting higher-order dynamical diagnostics. Generative approach potentially allows to generate ensemble of high-resolution reconstructions each of which preserves the fine-scale features.
2. Regression approach provides no robust way to transfer from training environment (simulation data) to real-world observations. Subtle differences between real observations at inference and simulated observations during training propagate through the network with no well-defined behaviour. Generative approach would ensure fields generated from observations 'look like' the simulated data - i.e. hopefully preserve the simulation's physics.

## **The Method**: [Score-Based Data Assimilation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html) (referred to here as 'generative data assimilation' or 'GenDA')

*Step 1:* Train unconditional diffusion model to produce realistic multi-modal samples. NB: this training is conducted on full model fields with no generation of simulated observations. <br>
![Alt text](images/GenDA training schematic v2.png)
*Step 2:* Guide the generation from the trained model using sparse observations by taking gradient steps wrt the state estimate, x, while keeping the diffusion model parameters fixed to preserve the qualitative nature of the model outputs. (Method proposed by [Rozet & Louppe 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html) and recently applied to atmospheric reanalysis by [Manshausen et al.](https://arxiv.org/abs/2406.16947)).
![Alt text](images/GenDA inference schematic v3.png)

**Training data**: simulation data from the 1/12 degree global reanalysis product [GLORYS 12](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) sub-setted in a region surrounding the Gulf Stream. 



**Experiments**:
1. Observing System Simulation Experiment (OSSE): estimate state from simulated satellite observations and compare to known 2D ground truth.
2. Observing System Experiment (OSE): estimate state from real-world satellite observations and compare to some independent withheld observations.

**Structure of the code:**

1. ```./pre-processing``` contains code for preparing the desired target fields from publicly available datasets. For example, we subtract geostrophic currents and Ekman currents (derived using a linear regression model) from the surface current variable we seek to reconstruct.
2. ```./src``` contains utility code (e.g. dataloaders, neural network architecture for a baseline UNet regression approach)
3. The GenDA diffusion model code is adapted from [NVIDIA Modulus CorrDiff](https://github.com/NVIDIA/modulus/tree/main/examples/generative/corrdiff)(installed from upstream repo on 07/21/2024, looks like they refactored the code since).
4. ```./conf``` contains hydra config files used for model training.
5. ```./sda``` contains the code for the score-based data assimilation method (i.e. observation-guided inference given a diffusion model trained using the CorrDiff code). This is a minor adaptation from [the orginal implementation](https://github.com/francois-rozet/sda) incorporating the modification described in Appendix B of [Manshausen et al.](https://arxiv.org/abs/2406.16947).
6. ```./training``` contains training scripts.
7. ```./inference``` contains inference scripts for both the OSE and OSSE.
8. ```./viz.ipynb``` visualizes the reconstructions.
9. More quantitative evaluation metrics coming soon...