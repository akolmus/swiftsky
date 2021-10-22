import math
import torch
import torch.nn as nn
import numpy as np

from scipy.interpolate import interp1d
from torch import Tensor
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_total_mass, total_mass_and_mass_ratio_to_component_masses

kappas = np.loadtxt('data/kappas.txt')
degsqr = np.loadtxt('data/degsqr.txt')
kappa2interp1d = interp1d(kappas, degsqr)


def kappa2sqrdeg(kappa: Tensor) -> Tensor:
    """ Convert the kappa constant to square degrees using a lookup table """
    return kappa2interp1d(kappa)


def prior2masses(chirp_mass, mass_ratio):
    """ Convert the chirp mass and mass ratio to m1 and m2 """
    total_mass = chirp_mass_and_mass_ratio_to_total_mass(chirp_mass, mass_ratio)
    m1, m2 = total_mass_and_mass_ratio_to_component_masses(mass_ratio, total_mass)

    if type(m1) == Tensor:
        return torch.stack((m1, m2), dim=1).to(torch.float32)
    else:
        return (m1, m2)


def decra2xyz(dec: torch.Tensor, ra: torch.Tensor) -> torch.Tensor:
    """ Calculate the Cartesian coordinates form the celestial coordinates """
    vec = torch.stack((torch.cos(dec) * torch.cos(ra), torch.cos(dec) * torch.sin(ra), torch.sin(dec)), dim=1)
    return vec.to(torch.float32)


def xyz2decra(xyz: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """ """
    dec = torch.atan2(xyz[:, 2], (xyz[:, 0] ** 2 + xyz[:, 1] ** 2) ** .5)
    ra = torch.remainder(torch.atan2(xyz[:, 1], xyz[:, 0]), 2 * math.pi)
    return dec, ra


def vmf_loss(kappa: Tensor, out: Tensor, tar: Tensor):
    """ The negative loglikelihood for the Von Mises Fisher distribution for p=3 """
    return -(torch.log(kappa + 1e-10) - torch.log(- torch.exp(-2 * kappa) + 1) - kappa - math.log(2 * math.pi) + (kappa * out * tar).sum(axis=1, keepdims=True))


def output2mu_sigma(output: Tensor, alpha: float = 5e-3, eps: float = 1e-3):
    """ Code taken adapted from paper https://arxiv.org/pdf/1910.14215.pdf, simplified to just fit dim = 2 """
    mean = output[:, :2]
    var = output[:, 2:4].clamp(max=44).exp()
    var_mat = torch.sqrt(var.unsqueeze(2) * var.unsqueeze(1))

    # Build correlation matrix
    rho_mat = torch.ones_like(var_mat)
    off_diag = torch.squeeze((1 - eps) * torch.tanh(alpha * output[:, 4:]))
    rho_mat[:, 0, 1] = off_diag
    rho_mat[:, 1, 0] = off_diag

    return mean, rho_mat * var_mat


def mvg_loss(out: Tensor, tar: Tensor):
    """ The negative loglikelihood for the multivariate Gaussian (2D in this case) """
    mean, covar = output2mu_sigma(out)
    err = (mean - tar).unsqueeze(-1)

    term1 = err * covar.inverse().bmm(err)
    term15 = covar.to('cpu:0').det().clamp(min=1e-10).to('cuda:0')
    term2 = torch.log(term15)
    loss = torch.mean(term1.sum(1) + term2, dim=-1) / 2 + math.log(2 * math.pi)

    return loss


class Normalizer(nn.Module):
    """ Learns the shift and variance of a bunch data """

    def __init__(self, use_mean: bool = True, use_scale: bool = True, num_channels: int = 3):
        """  """
        super(Normalizer, self).__init__()
        self.use_mean = use_mean
        self.use_scale = use_scale
        self.std = nn.Parameter(torch.ones(size=(1, num_channels, 1), requires_grad=False))
        self.mean = nn.Parameter(torch.zeros(size=(1, num_channels, 1), requires_grad=False))

    def set_meanvariance(self, batch: torch.Tensor):
        """ Set the scale and mean shift """
        std, mean = torch.std_mean(batch, dim=[0, 2], unbiased=True, keepdim=True)
        self.std = nn.Parameter(std, requires_grad=False)
        self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mean:
            x = x - self.mean
        if self.use_scale:
            x = x / self.std
        return x


class RunningAverageMeter(object):
    """ Computes and stores the average and current value - taken from the CNF example in the torch diffeq repo """

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class EpochAverageMeter(object):
    """ Computes the average of the loss over the epoch used for the validation run """

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.seen_value = 0
        self.seen_samples = 0

    def update(self, val, size):
        self.seen_value += val * size
        self.seen_samples += size
        self.avg = self.seen_value / self.seen_samples

