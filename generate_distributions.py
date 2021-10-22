import argparse
import warnings
import logging
import math
import torch
import pickle
import numpy as np
import utils_train

from data.gwave import GravitationalWaveDataset
from neural.net import CNN

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Setup the argument parser
parser = argparse.ArgumentParser("Train a continuous normalizing flow model")

# General arguments
parser.add_argument("--name", type=str, default="with_sky_values")
parser.add_argument("--warnings", type=bool, default=True)
parser.add_argument("--identifier", type=str, default="with_mass_norm_warm_restarts")

# Data arguments
parser.add_argument("--H1_active", type=bool, default=True)
parser.add_argument("--L1_active", type=bool, default=True)
parser.add_argument("--V1_active", type=bool, default=True)
parser.add_argument("--sampling_frequency", type=int, default=2048)
parser.add_argument("--merger_position", type=float, default=0.5)
parser.add_argument("--initial_duration", type=float, default=2.0)
parser.add_argument("--cut_duration", type=float, default=2.0)
parser.add_argument("--waveform_approximant", type=str, default="IMRPhenomPv2", choices=["IMRPhenomPv2"])
parser.add_argument("--reference_frequency", type=float, default=50.0)
parser.add_argument("--minimum_frequency", type=float, default=20.0)
parser.add_argument("--maximum_frequency", type=float, default=2048.0)
parser.add_argument("--whiten_time_series", type=bool, default=True)
parser.add_argument("--inject_noise", type=bool, default=True)
parser.add_argument("--snr_min", type=float, default=10.0)  # This is here for the __init__ of the dataset, we change later
parser.add_argument("--snr_max", type=float, default=100.0) # This is here for the __init__ of the dataset, we change later
parser.add_argument("--snr_peak", type=float, default=15.0) # This is here for the __init__ of the dataset, we change later
parser.add_argument("--snr_temp", type=float, default=15.0) # This is here for the __init__ of the dataset, we change later

# Prior arguments
parser.add_argument("--chirp_mass_minimum", type=float, default=10.0)
parser.add_argument("--chirp_mass_maximum", type=float, default=100.0)
parser.add_argument("--mass_ratio_minimum", type=float, default=0.25)
parser.add_argument("--mass_ratio_maximum", type=float, default=1.000)
parser.add_argument("--mass_minimum", type=float, default=20.0)
parser.add_argument("--mass_maximum", type=float, default=80.0)
parser.add_argument("--aligned_spin", type=bool, default=True)
parser.add_argument("--spin_minimum", type=float, default=0.0)
parser.add_argument("--spin_maximum", type=float, default=0.0 + 1e-10)
parser.add_argument("--phi_12", type=bool, default=True)
parser.add_argument("--phi_jl", type=bool, default=True)
parser.add_argument("--theta_jn", type=bool, default=True)
parser.add_argument("--psi", type=bool, default=True)
parser.add_argument("--phase", type=bool, default=True)
parser.add_argument("--geocent_time", type=bool, default=False)

# Training arguments
parser.add_argument("--val_samples", type=int, default=10000)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu:0'])
parser.add_argument("--benchmark_cudnn", type=bool, default=True)

# SNR range
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--snr", type=str, default="10, 15, 20")


if __name__ == '__main__':
    # Given arguments
    args = parser.parse_args()
    path = f'model/{args.identifier}'

    # Create directory or load the previous session
    if not Path(path).exists():
        Path(path).mkdir(exist_ok=True, parents=True)

    # Ignore the RunTime Warning for the Bilby library.
    if args.warnings:
        logging.getLogger('bilby').setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Setup the device
    device = torch.device(args.device)
    if args.benchmark_cudnn: torch.backends.cudnn.benchmark = True

    # Setup the seed
    np.random.seed(args.seed)

    # Setup the data
    _data = GravitationalWaveDataset(args, train=False)
    _loader = DataLoader(_data,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         pin_memory=True,
                         drop_last=False,
                         prefetch_factor=2,
                         worker_init_fn=lambda _: np.random.seed(args.seed))

    # Setup model
    model = CNN(3)
    model = model.to(device)

    # Setup the normalizer
    normalizer = utils_train.Normalizer(use_mean=True, use_scale=True, num_channels=int(args.H1_active + args.L1_active + args.V1_active))
    if Path(f"{path}/normalizer_checkpoint.pth").exists():
        normalizer.load_state_dict(torch.load(f"{path}/normalizer_checkpoint.pth"))

    # Load the model
    model_checkpoints = [str(f) for f in Path(f'{path}/').glob("checkpoint_*")]
    if model_checkpoints:
        # Get best checkpoint
        bst_model_checkpoint = sorted(model_checkpoints, key=lambda x: float(x.split(f"checkpoint_")[1].replace('.pth', '')))[0]
        checkpoint = torch.load(bst_model_checkpoint)

        # Set the correct values
        model.load_state_dict(checkpoint['model_state_dict'])

    try:
        for snr_peak in list(map(int, args.snr.split(","))):
            print(snr_peak)
            _data.set_snr_values(snr_min=snr_peak, snr_max=snr_peak, snr_peak=snr_peak, snr_temp=15)

            with torch.no_grad():
                model.train(mode=False)
                for batch_idx, (gwave, para) in enumerate(_loader):
                    gwave = normalizer(gwave).to(device)
                    cart_celestial = utils_train.decra2xyz(para['dec'], para['ra']).to(device)
                    kappa, out_celestial, out_mass_mean, out_mass_cov = model.forward(gwave)

                    for i in range(gwave.shape[0]):
                        with open(f'samples/neural_data_{args.name}_{snr_peak}_{i}.pck', 'wb') as f:
                            summary = {"wave": gwave[i].detach().cpu().numpy(),
                                       "para": {k: v[i].numpy() for k, v in para.items()},
                                       "kappa": kappa[i].detach().cpu(),
                                       "out_celestial": out_celestial[i].detach().cpu(),
                                       "out_mass_mean": out_mass_mean[i].detach().cpu(),
                                       "out_mass_cov": out_mass_cov[i].detach().cpu()}
                            pickle.dump(summary, f)

    except KeyboardInterrupt():
        pass
