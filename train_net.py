import argparse
import warnings
import logging
import math
import json
import torch
import torch.optim as optim
import numpy as np
import utils_train

from data.gwave import GravitationalWaveDataset
from neural.net import CNN

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


# Setup the argument parser
parser = argparse.ArgumentParser("Train a neural network to infer the sky and masses from a GW")

# General arguments
parser.add_argument("--warnings", type=bool, default=True)
parser.add_argument("--identifier", type=str, default="warm_restarts")

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
parser.add_argument("--snr_min", type=float, default=10.0)
parser.add_argument("--snr_max", type=float, default=100.0)
parser.add_argument("--snr_peak", type=float, default=15.0)
parser.add_argument("--snr_temp", type=float, default=15.0)

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
parser.add_argument("--trn_samples", type=int, default=500000)
parser.add_argument("--val_samples", type=int, default=100000)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu:0'])
parser.add_argument("--benchmark_cudnn", type=bool, default=True)


if __name__ == '__main__':
    # Given arguments
    args = parser.parse_args()
    path = f'model/{args.identifier}'

    # Create directory or load the previous session
    if not Path(path).exists():
        Path(path).mkdir(exist_ok=True, parents=True)
    if not Path(f'{path}/arguments.txt').exists():
        with open(f'{path}/arguments.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    print(f"Given arguments:")
    for arg in vars(args):
        num_spaces = (32 - len(arg)) * ' '
        print(f"{arg}:{num_spaces}{getattr(args, arg)}")

    # Ignore the RunTime Warning for the Bilby library.
    if args.warnings:
        logging.getLogger('bilby').setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Setup the device
    device = torch.device(args.device)
    if args.benchmark_cudnn: torch.backends.cudnn.benchmark = True

    # Setup the data
    trn_data = GravitationalWaveDataset(args, train=True)
    val_data = GravitationalWaveDataset(args, train=False)
    trn_loader = DataLoader(trn_data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2,
                            worker_init_fn=lambda _: np.random.seed(None))
    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False,
                            prefetch_factor=2,
                            worker_init_fn=lambda _: np.random.seed(None))

    # Setup model
    model = CNN(3)
    model = model.to(device)

    # Setup optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-5)

    # Setup the normalizer
    normalizer = utils_train.Normalizer(use_mean=True, use_scale=True, num_channels=int(args.H1_active + args.L1_active + args.V1_active))
    if Path(f"{path}/normalizer_checkpoint.pth").exists():
        normalizer.load_state_dict(torch.load(f"{path}/normalizer_checkpoint.pth"))
    else:
        normalizer.set_meanvariance(trn_data.batch_noise(num_samples=1000))
        torch.save(normalizer.state_dict(), f"{path}/normalizer_checkpoint.pth")

    # Setup the Running Average Meter
    trn_vmf_loss_meter = utils_train.RunningAverageMeter()
    trn_mvg_loss_meter = utils_train.RunningAverageMeter()
    val_vmf_loss_meter = utils_train.EpochAverageMeter()
    val_mvg_loss_meter = utils_train.EpochAverageMeter()
    val_kappa_meter = utils_train.EpochAverageMeter()
    val_angle_meter = utils_train.EpochAverageMeter()

    # Load the model
    model_checkpoints = [str(f) for f in Path(f'{path}/').glob("checkpoint_*")]
    start_epoch = 0
    if model_checkpoints:
        # Get best checkpoint
        bst_model_checkpoint = sorted(model_checkpoints, key=lambda x: float(x.split(f"checkpoint_")[1].replace('.pth', '')))[0]
        checkpoint = torch.load(bst_model_checkpoint)

        # Set the correct values
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    try:
        for epoch_nb in range(start_epoch, args.num_epochs):
            # Reset the metric trackers
            trn_vmf_loss_meter.reset()
            trn_mvg_loss_meter.reset()
            val_vmf_loss_meter.reset()
            val_mvg_loss_meter.reset()
            val_kappa_meter.reset()
            val_angle_meter.reset()

            model.train()
            loop = tqdm(trn_loader)
            for gwave, para in loop:
                gwave = normalizer(gwave).to(device)
                cart_celestial = utils_train.decra2xyz(para['dec'], para['ra']).to(device)
                mass_means = utils_train.prior2masses(para['chirp_mass'], para['mass_ratio']).to(device)
                mass_means = (mass_means - 30) / 50
                optimizer.zero_grad()

                # Forward pass
                kappa, out_celestial, out_mass_mean, out_mass_cov = model.forward(gwave)
                loss_vmf = utils_train.vmf_loss(kappa, out_celestial, cart_celestial)
                loss_mvg = utils_train.mvg_loss(torch.cat([out_mass_mean, out_mass_cov], dim=1), mass_means)
                loss = torch.add(loss_vmf.mean(), loss_mvg.mean())

                # Backward
                loss.backward()
                optimizer.step()

                # Metrics
                trn_vmf_loss_meter.update(loss_vmf.detach().mean().cpu().numpy())
                trn_mvg_loss_meter.update(loss_mvg.detach().mean().cpu().numpy())

                # Onscreen update
                loop.set_description(f"Epoch: {epoch_nb}/{args.num_epochs} - trn")
                loop.set_postfix_str(s=f"VMF loss: {trn_vmf_loss_meter.avg:.3f} - current VMF loss: {loss_vmf.mean().item():.3f} - MVG loss: {trn_mvg_loss_meter.avg:.3f} - current MVG loss: {loss_mvg.mean().item():.3f}")

            with torch.no_grad():
                model.train(mode=False)
                loop = tqdm(val_loader)
                for gwave, para in loop:
                    gwave = normalizer(gwave).to(device)
                    cart_celestial = utils_train.decra2xyz(para['dec'], para['ra']).to(device)
                    mass_means = utils_train.prior2masses(para['chirp_mass'], para['mass_ratio']).to(device)
                    mass_means = (mass_means -30) / 50
                    optimizer.zero_grad()

                    # Forward pass
                    kappa, out_celestial, out_mass_mean, out_mass_cov = model.forward(gwave)
                    loss_vmf = utils_train.vmf_loss(kappa, out_celestial, cart_celestial)
                    loss_mvg = utils_train.mvg_loss(torch.cat([out_mass_mean, out_mass_cov], dim=1), mass_means)

                    # Metrics
                    val_vmf_loss_meter.update(loss_vmf.detach().mean().cpu().numpy(), size=gwave.shape[0])
                    val_mvg_loss_meter.update(loss_mvg.detach().mean().cpu().numpy(), size=gwave.shape[0])
                    val_kappa_meter.update(kappa.detach().mean().cpu().numpy(), size=gwave.shape[0])
                    val_angle_meter.update((torch.acos((out_celestial * cart_celestial).sum(axis=1, keepdims=True)) * 180 / math.pi).detach().mean().cpu().numpy(), size=gwave.shape[0])

                    # Onscreen update
                    loop.set_description(f"Epoch: {epoch_nb}/{args.num_epochs} - val")
                    loop.set_postfix_str(s=f"VMF loss: {val_vmf_loss_meter.avg:.3f} - MVG loss: {val_mvg_loss_meter.avg:.3f} - avg_kappa: {val_kappa_meter.avg:.3f} - avg_angle: {val_angle_meter.avg:.3f}")

            # Finish the current stuff
            scheduler.step()

            # Save model
            torch.save(obj={'epoch': epoch_nb,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()},
                       f=f'{path}/checkpoint_{val_vmf_loss_meter.avg:.3f}.pth')

    except KeyboardInterrupt:
            pass
