import torch
import torch.nn as nn

from collections import OrderedDict
from torch import Tensor


class CNN(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()

        layers = OrderedDict([
            ('conv0', nn.Conv1d(in_channels, 16, kernel_size=16, dilation=1)),
            ('actf0', nn.ReLU()),
            ('maxp0', nn.MaxPool1d((2,))),

            ('conv1', nn.Conv1d(16, 16, kernel_size=64, dilation=1)),
            ('actf1', nn.ReLU()),
            ('maxp1', nn.MaxPool1d((2,))),

            ('conv2', nn.Conv1d(16, 32, kernel_size=64, dilation=2)),
            ('actf2', nn.ReLU()),
            ('maxp2', nn.MaxPool1d((4,))),

            ('conv3', nn.Conv1d(32, 32, kernel_size=64, dilation=2)),
            ('actf3', nn.ReLU()),

            ('flatten', nn.Flatten()),
        ])

        self.conv = nn.Sequential(layers)
        self.linear_celestial = nn.Sequential(OrderedDict([
            ('linear0', nn.Linear(in_features=2848, out_features=256)),
            ('batchn0', nn.BatchNorm1d(num_features=256)),
            ('actf0', nn.LeakyReLU(0.1, inplace=True)),
            ('linear1', nn.Linear(in_features=256, out_features=3))
        ]))

        self.linear_mass_mean = nn.Sequential(OrderedDict([
            ('linear0', nn.Linear(in_features=2848, out_features=256)),
            ('batchn0', nn.BatchNorm1d(num_features=256)),
            ('actf0', nn.LeakyReLU(0.1, inplace=True)),
            ('linear1', nn.Linear(in_features=256, out_features=2)),
        ]))

        self.linear_mass_cov = nn.Sequential(OrderedDict([
            ('linear0', nn.Linear(in_features=2848, out_features=256)),
            ('batchn0', nn.BatchNorm1d(num_features=256)),
            ('actf0', nn.LeakyReLU(0.1, inplace=True)),
            ('linear1', nn.Linear(in_features=256, out_features=3))
        ]))

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        """ Forward pass through the network """
        x = self.conv(x)

        out_celestial = self.linear_celestial(x)
        out_mass_mean = self.linear_mass_mean(x)
        out_mass_cov = self.linear_mass_cov(x)

        kappa = torch.norm(out_celestial, p=2, dim=-1).view(-1, 1)
        out_celestial = 1 / kappa * out_celestial

        return kappa, out_celestial, out_mass_mean, out_mass_cov


