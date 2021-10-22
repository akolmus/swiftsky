import numpy as np
import torch
import math

from torch import Tensor
from torch.distributions import Beta, Uniform


class VMFDistribution(object):
    """ A class object for the Von Mises Fisher distribution """

    def __init__(self, kappa: Tensor, mu: Tensor):
        """ """
        assert len(kappa.shape) == 1 and len(mu.shape) == 1 and mu.shape[0] == 3, "Given kappa and mu do not have the correct dimensionality"
        self.kappa, self.mu = kappa, mu

    def pdf(self, vec: Tensor):
        """ The probability density for given samples in vec assuming [N, [x, y, z]] format for vec """
        assert torch.norm(vec, dim=1).all() == 1
        assert len(vec.shape) == 2

        if self.kappa > 100:
            return self.kappa / (2 * math.pi * (1 - torch.exp(-2 * self.kappa))) * torch.exp(self.kappa * (vec * self.mu.unsqueeze(0)).sum(axis=1, keepdims=True) - 1)
        else:
            return self.kappa / (2 * math.pi * (torch.exp(self.kappa) - torch.exp(-self.kappa))) * torch.exp(self.kappa * (vec * self.mu.unsqueeze(0)).sum(axis=1, keepdims=True))

    def logpdf(self, vec: Tensor):
        """ The logprobability density for given samples in vec assuming [N, [x, y, z]] format for vec """
        assert torch.norm(vec, dim=1).all() == 1
        assert len(vec.shape) == 2

        return torch.log(self.kappa + 1e-10) - torch.log(1 - torch.exp(-2 * self.kappa)) - self.kappa - math.log(2 * math.pi) + self.kappa * (vec * self.mu.unsqueeze(0)).sum(axis=1, keepdims=True)

    def sample(self, num_samples: int):
        """ Generate random samples from distribution using rejection sampling, based on https://math.stackexchange.com/questions/1326492/sampling-from-the-von-mises-fisher-distribution """

        # Setup a few constants
        dim = 3 - 1
        sqrt = torch.sqrt(4 * self.kappa ** 2 + dim ** 2)
        envelop_param = (-2 * self.kappa + sqrt) / dim
        node = (1. - envelop_param) / (1. + envelop_param)
        correction = self.kappa * node + dim * torch.log(1. - node ** 2)

        # # Setup base distributions
        beta_dist = Beta(dim / 2, dim / 2)
        unif_dist = Uniform(0, 1)

        # Apply rejection sampling
        counter = 0
        coord_x = torch.zeros(num_samples)
        while counter < num_samples:
            sym_beta = beta_dist.sample((num_samples - counter,))
            sym_unif = unif_dist.sample((num_samples - counter,))

            sym_coord_x = (1 - (1 + envelop_param) * sym_beta) / (1 - (1 - envelop_param) * sym_beta)
            acc_coord_x = torch.masked_select(sym_coord_x, (self.kappa * sym_coord_x + dim * torch.log(1 - node * sym_coord_x) - correction) >= torch.log(sym_unif))

            coord_x[counter:(counter + acc_coord_x.shape[0])] = acc_coord_x
            counter += acc_coord_x.shape[0]

        # Obtain the other coordinates, need to be orthogonal
        coord_other = torch.randn(size=(num_samples, 2))
        coord_other = coord_other / torch.norm(coord_other, dim=1, keepdim=True)
        coord_other = torch.einsum('...,...i->...i', torch.sqrt(1 - coord_x ** 2), coord_other)

        # Assemble a vector based on mu being [1, 0, 0]
        vec = torch.cat((coord_x.unsqueeze(1), coord_other), dim=1)

        # Calculate rotation matrix to align samples with the actual mu
        embd = torch.cat((self.mu[None, :], torch.zeros((2, 3))))
        norm = torch.norm(embd)
        q, _ = torch.linalg.qr(embd.T / norm)

        # print(torch.mean(torch.acos((self.mu * torch.matmul(vec[None, :], q.T).squeeze()).sum(axis=1, keepdims=True)) * 180 / math.pi))

        # Pytorch implementation of QR decomposition is not always consistent, need to fix that here.
        if torch.mean(torch.acos((self.mu * torch.matmul(vec[None, :], q.T).squeeze()).sum(axis=1, keepdims=True)) * 180 / math.pi) > 90:
            vec = -torch.matmul(vec[None, :], q.T)
        else:
            vec = torch.matmul(vec[None, :], q.T)

        return vec.squeeze()


if __name__ == '__main__':

    vmf = VMFDistribution(kappa=torch.Tensor([0.1]), mu=torch.Tensor([1 / (2 ** .5), 0, 1 / (2 ** .5)]))
    print(vmf.sample(1000))

    # import healpy as hp
    # import time
    # kappas = np.hstack((np.arange(0.01, 1, 0.01), np.arange(1, 10, 0.1), np.arange(10, 100, 0.5), np.arange(100, 500, 1), np.arange(500, 1000, 2)))
    # degsqr = np.zeros_like(kappas)
    # np.savetxt('data/kappas.txt', kappas)
    # for idx, kappa in enumerate(kappas):
    #     vmf = VMFDistribution(kappa=torch.Tensor([kappa]), mu=torch.Tensor([0, 1 / (2 ** .5), -1 / (2 ** .5)]))
    #     num_side = 64
    #     num_samples = 100000
    #     pixel = torch.zeros(hp.nside2npix(num_side))
    #
    #     t0 = time.time()
    #     for sample in vmf.sample(num_samples):
    #         pixel[hp.vec2pix(num_side, sample[0], sample[1], sample[2], nest=True)] += 1
    #
    #     pixel_value, pixel_idx = torch.sort(pixel, descending=True)
    #     pixel_cumul = torch.cumsum(pixel_value, dim=0)
    #     pixel_req = sum([1 for v in pixel_cumul if v < .9 * num_samples])
    #     degsqr[idx] = pixel_req * hp.nside2pixarea(nside=num_side, degrees=True)
    #
    #     print(f"kappa: {vmf.kappa[0]:.2f}: 90th percentile deg^2: {pixel_req * hp.nside2pixarea(nside=num_side, degrees=True):.3f}")
    #     print(time.time() - t0)
    #
    # np.savetxt('data/degsqr.txt', degsqr)
