import argparse
import warnings
import logging
import math
import torch
import numpy as np
import utils_train
import matplotlib.pyplot as plt

from data.gwave import GravitationalWaveDataset
from neural.net import CNN

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


# Setup the argument parser
parser = argparse.ArgumentParser("Train a continuous normalizing flow model")

# General arguments
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
parser.add_argument("--aligned_spin", type=bool, default=False)
parser.add_argument("--spin_minimum", type=float, default=0.0)
parser.add_argument("--spin_maximum", type=float, default=0.95)
parser.add_argument("--tilt1", type=bool, default=True)
parser.add_argument("--tilt2", type=bool, default=True)
parser.add_argument("--phi_12", type=bool, default=True)
parser.add_argument("--phi_jl", type=bool, default=True)
parser.add_argument("--theta_jn", type=bool, default=True)
parser.add_argument("--psi", type=bool, default=True)
parser.add_argument("--phase", type=bool, default=True)
parser.add_argument("--geocent_time", type=bool, default=True)

# Training arguments
parser.add_argument("--val_samples", type=int, default=10000)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu:0'])
parser.add_argument("--benchmark_cudnn", type=bool, default=True)

# SNR range
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--snr_initial", type=float, default=5.0)
parser.add_argument("--snr_closing", type=float, default=50.0)
parser.add_argument("--snr_stepsize", type=float, default=0.5)


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

    # Setup the Running Average Meter
    _vmf_loss_meter = utils_train.EpochAverageMeter()
    _mvg_loss_meter = utils_train.EpochAverageMeter()
    _kappa_meter = utils_train.EpochAverageMeter()
    _angle_meter = utils_train.EpochAverageMeter()
    _mass_mae_meter = utils_train.EpochAverageMeter()
    _mass_mre_meter = utils_train.EpochAverageMeter()

    # For visualization
    snr_range = np.arange(start=args.snr_initial, stop=args.snr_closing + args.snr_stepsize, step=args.snr_stepsize)
    maae = np.zeros_like(snr_range)
    deg = np.zeros_like(snr_range)
    rme = np.zeros_like(snr_range)

    try:
        for idx, snr_peak in enumerate(snr_range):
            _data.set_snr_values(snr_min=snr_peak, snr_max=snr_peak, snr_peak=snr_peak, snr_temp=15)
            _vmf_loss_meter.reset()
            _mvg_loss_meter.reset()
            _kappa_meter.reset()
            _angle_meter.reset()
            _mass_mae_meter.reset()
            _mass_mre_meter.reset()

            with torch.no_grad():
                model.train(mode=False)
                loop = tqdm(_loader)
                for gwave, para in loop:
                    gwave = normalizer(gwave).to(device)
                    cart_celestial = utils_train.decra2xyz(para['dec'], para['ra']).to(device)
                    mass_means = utils_train.prior2masses(para['chirp_mass'], para['mass_ratio']).to(device)
                    mass_means = (mass_means - 30) / 50

                    # Forward pass
                    kappa, out_celestial, out_mass_mean, out_mass_cov = model.forward(gwave)
                    loss_vmf = utils_train.vmf_loss(kappa, out_celestial, cart_celestial)
                    loss_mvg = utils_train.mvg_loss(torch.cat([out_mass_mean, out_mass_cov], dim=1), mass_means)

                    # Metrics
                    _vmf_loss_meter.update(loss_vmf.detach().mean().cpu().numpy(), size=gwave.shape[0])
                    _mvg_loss_meter.update(loss_mvg.detach().mean().cpu().numpy(), size=gwave.shape[0])
                    _kappa_meter.update(kappa.detach().mean().cpu().numpy(), size=gwave.shape[0])
                    _angle_meter.update((torch.acos((out_celestial * cart_celestial).sum(axis=1, keepdims=True)) * 180 / math.pi).detach().mean().cpu().numpy(), size=gwave.shape[0])
                    _mass_mae_meter.update(50 * (out_mass_mean - mass_means).detach().abs().mean().cpu().numpy(), size=gwave.shape[0])
                    _mass_mre_meter.update((1 - (50 * out_mass_mean + 30) / (50 * mass_means + 30)).detach().abs().mean().cpu().numpy(), size=gwave.shape[0])

                    # Onscreen update
                    loop.set_description(f"SNR: {snr_peak:.1f} -")
                    loop.set_postfix_str(s=f"VMF loss: {_vmf_loss_meter.avg:.3f} - MVG loss: {_mvg_loss_meter.avg:.3f} - avg_kappa: {_kappa_meter.avg:.3f} - avg_angle: {_angle_meter.avg:.3f} - deg_sqr: {utils_train.kappa2sqrdeg(_kappa_meter.avg):.3f} - mae_mass: {_mass_mae_meter.avg:.3f} - mre_mass: {_mass_mre_meter.avg:.3f}")

            maae[idx] = _angle_meter.avg
            deg[idx] = utils_train.kappa2sqrdeg(_kappa_meter.avg)
            rme[idx] = _mass_mre_meter.avg

    except KeyboardInterrupt:
        pass

    plt.style.use('seaborn-pastel')

    plt.plot(snr_range, maae)
    plt.ylabel('maae($^\circ$)')
    plt.xlabel('Optimal SNR')
    plt.ylim([0, 85])
    plt.xlim(xmin=args.snr_initial, xmax=args.snr_closing)
    plt.show()

    plt.semilogy(snr_range, deg)
    plt.ylabel('90% confidence area ($^{\circ ^2})$')
    plt.xlabel('Optimal SNR')
    plt.ylim(ymin=100)
    plt.xlim(xmin=args.snr_initial, xmax=args.snr_closing)
    plt.show()

    plt.plot(snr_range, rme)
    plt.ylabel('mean relative error mass')
    plt.xlabel('Optimal SNR')
    plt.ylim(ymin=0)
    plt.xlim(xmin=args.snr_initial, xmax=args.snr_closing)
    plt.show()