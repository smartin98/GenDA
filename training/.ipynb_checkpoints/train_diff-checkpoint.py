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


#################
# NVIDIA Modulus CorrDiff code with minor adaptations from Scott Martin to apply to surface ocean state estimation
#################

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""
import json
import os
import shutil
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '55000'
tmp_dir = '/nobackup/samart18/tmp'
os.environ['TMPDIR'] = tmp_dir
import sys
sys.path.append('/nobackup/samart18/edm')
sys.path.append('/nobackup/samart18/modulus')
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
import numpy as np
from datetime import date, timedelta

import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf, DictConfig, ListConfig

sys.path.append('src')
from src.dataloaders import *
import math
import csv

import modulus
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import EasyDict, parse_int_list, InfiniteSampler

from training_diff import training_loop
# from datasets.dataset import init_train_valid_datasets_from_config

def IQR(dist):
    return np.percentile(dist, 75) - np.percentile(dist, 25)


@hydra.main(version_base="1.2", config_path="conf", config_name="config_train_base")
def main(cfg: DictConfig) -> None:
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    
    # Initialize distributed manager.
    DistributedManager.initialize()
    dist = DistributedManager()

    # Sanity check
    if not hasattr(cfg, "task"):
        raise ValueError(
            """Need to specify the task. Make sure the right config file is used. Run training using python train.py --config-name=<your_yaml_file>.
            For example, for regression training, run python train.py --config-name=config_train_regression.
            And for diffusion training, run python train.py --config-name=config_train_diffusion."""
        )

    # Dump the configs
    os.makedirs(cfg.outdir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.outdir, "config.yaml"))

    task = getattr(cfg, "task")
    outdir = getattr(cfg, "outdir", "./output")
    arch = getattr(cfg, "arch", "ddpmpp-cwb-v0-regression")
    precond = getattr(cfg, "precond", "unetregression")

    duration = getattr(cfg, "duration", 200)
    batch_size_global = getattr(cfg, "batch_size_global", 256)
    batch_size_gpu = getattr(cfg, "batch_size_gpu", 2)
    cbase = getattr(cfg, "cbase", 1)
    # cres = parse_int_list(getattr(cfg, "cres", None))
    lr = getattr(cfg, "lr", 0.0002)
    ema = getattr(cfg, "ema", 0.5)
    dropout = getattr(cfg, "dropout", 0.13)
    augment = getattr(cfg, "augment", 0.0)
    physics_loss = getattr(cfg, "physics_loss", False)
    

    # Parse performance options
    if hasattr(cfg, "fp_optimizations"):
        fp_optimizations = cfg.fp_optimizations
        fp16 = fp_optimizations == "fp16"
    else:
        # look for legacy "fp16" parameter
        fp16 = getattr(cfg, "fp16", False)
        fp_optimizations = "fp16" if fp16 else "fp32"
    ls = getattr(cfg, "ls", 1)
    bench = getattr(cfg, "bench", True)
    workers = getattr(cfg, "workers", 4)
    songunet_checkpoint_level = getattr(cfg, "songunet_checkpoint_level", 0)

    tick = getattr(cfg, "tick", 1)
    dump = getattr(cfg, "dump", 500)
    validation_dump = getattr(cfg, "validation_dump", 500)
    validation_steps = getattr(cfg, "validation_steps", 10)
    seed = getattr(cfg, "seed", 0)
    dry_run = getattr(cfg, "dry_run", False)
    
    with open('/nobackup/samart18/GenDA/input_data/diffusion_training_rescale_factors.json', 'r') as f:
        rescale_factors = json.load(f)

    
    buffers = 12
    data_dir = '../input_data/'
    ds = xr.open_dataset(data_dir + 'glorys_pre_processed_fixed_noislands.nc').astype('float32') # 'cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_70.00W-40.00W_25.00N-45.00N_0.49$
    ds_m = xr.open_dataset(data_dir + 'glorys_means_pre_processed_fixed_noislands.nc').astype('float32')
    ds_clim = xr.open_dataset(data_dir + 'glorys_gulfstream_climatology.nc').astype('float32').isel(depth = 0, drop = True)
    
    
    dataset = Diffusion_Training_Dataset(data_dir = '/nobackup/samart18/GenDA/input_data/', 
                                         latent_dim = 1, 
                                         n_lon = 128, 
                                         n_lat = 128, 
                                         date_range = [date(2010,1,1),date(2016,12,31)], 
                                         variables = ['zos', 'thetao', 'so', 'u_ageo_eddy', 'v_ageo_eddy', 'uas', 'vas'], 
                                         var_stds = rescale_factors, 
                                         lon_buffers = [buffers, buffers], 
                                         lat_buffers = [buffers, buffers + 6], 
                                         multiprocessing = False, 
                                         augment = False)
    
    batch_size = batch_size_global
    n_cpus = workers

    dataset_sampler = InfiniteSampler(
        dataset=dataset, rank=dist.rank, num_replicas=dist.world_size, seed=seed
    )
    
    dataset_iter = iter(DataLoader(dataset, sampler = dataset_sampler, batch_size=batch_size))

    dataset_val = Diffusion_Training_Dataset(data_dir = '/nobackup/samart18/GenDA/input_data/', 
                                         latent_dim = 1, 
                                         n_lon = 128, 
                                         n_lat = 128, 
                                         date_range = [date(2018,1,1),date(2020,12,31)], 
                                         variables = ['zos', 'thetao', 'so', 'u_ageo_eddy', 'v_ageo_eddy', 'uas', 'vas'], 
                                         var_stds = rescale_factors, 
                                         lon_buffers = [buffers, buffers], 
                                         lat_buffers = [buffers, buffers + 6], 
                                         multiprocessing = False, 
                                         augment = False)
    batch_size = batch_size_global
    n_cpus = workers

    dataset_val_sampler = InfiniteSampler(
        dataset=dataset_val, rank=dist.rank, num_replicas=dist.world_size, seed=seed
    )
    
    valid_dataset_iter = iter(DataLoader(dataset_val, sampler = dataset_val_sampler, batch_size=batch_size))
    
    c = EasyDict()
    c.task = task
    c.fp_optimizations = fp_optimizations
    c.grad_clip_threshold = getattr(cfg, "grad_clip_threshold", None)
    c.lr_decay = getattr(cfg, "lr_decay", 0.8)
    

    # Initialize logger.
    os.makedirs("logs", exist_ok=True)
    logger = PythonLogger(name="train")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging(file_name=f"logs/train_{dist.rank}.log")

    # inform about the output
    logger.info(
        f"Checkpoints, logs, configs, and stats will be written in this directory: {os.getcwd()}"
    )

    # Initialize config dict.
    c.network_kwargs = EasyDict()
    c.loss_kwargs = EasyDict()
    c.optimizer_kwargs = EasyDict(
        class_name="torch.optim.Adam", lr=lr, betas=[0.9, 0.999], eps=1e-8
    )

    
    c.network_kwargs.update(
        model_type="SongUNet",
        model_channels = 64,
        num_blocks = 2,
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        checkpoint_level=songunet_checkpoint_level,
    )
    
    
    c.network_kwargs.class_name = "modulus.models.diffusion.EDMPrecond"
    c.loss_kwargs.class_name = "modulus.metrics.diffusion.EDMLoss"
    
    if augment:
        if augment < 0 or augment > 1:
            raise ValueError("Augment probability should be within [0,1].")
        # import augmentation pipe
        try:
            from edmss import AugmentPipe
        except ImportError:
            raise ImportError(
                "Please get the augmentation pipe  by running: pip install git+https://github.com/mnabian/edmss.git"
            )
        c.augment_kwargs = EasyDict(class_name="edmss.AugmentPipe", p=augment)
        c.augment_kwargs.update(
            xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
        )
        c.network_kwargs.augment_dim = 9
        
    c.network_kwargs.update(dropout=dropout, use_fp16=fp16)

    # Training options.
    c.total_kimg = max(int(duration * 1000), 1)
    c.ema_halflife_kimg = int(ema * 1000)
    c.update(batch_size_gpu=batch_size_gpu, batch_size_global=batch_size_global)
    c.update(loss_scaling=ls, cudnn_benchmark=bench)
    c.update(
        kimg_per_tick=tick,
        state_dump_ticks=dump,
        valid_dump_ticks=validation_dump,
        num_validation_evals=validation_steps,
    )

    # Random seed.
    if seed is None:
        seed = torch.randint(1 << 31, size=[], device=dist.device)
        if dist.distributed:
            torch.distributed.broadcast(seed, src=0)
        seed = int(seed)

    c.run_dir = outdir

    # Print options.
    for key in list(c.keys()):
        val = c[key]
        if isinstance(val, (ListConfig, DictConfig)):
            c[key] = OmegaConf.to_container(val, resolve=True)
    logger0.info("Training options:")
    logger0.info(json.dumps(c, indent=2))
    logger0.info(f"Output directory:        {c.run_dir}")
    logger0.info(f"Network architecture:    {arch}")
    logger0.info(f"Preconditioning & loss:  {precond}")
    logger0.info(f"Number of GPUs:          {dist.world_size}")
    logger0.info(f"Batch size:              {c.batch_size_global}")
    logger0.info(f"Mixed-precision:         {c.fp_optimizations}")

    # Dry run?
    if dry_run:
        logger0.info("Dry run; exiting.")
        return

    # Create output directory.
    logger0.info("Creating output directory...")
    if dist.rank == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)

    # Train.
    training_loop.training_loop(
        dataset, dataset_iter, dataset_val, valid_dataset_iter, **c
    )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
