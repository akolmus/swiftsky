import argparse
import warnings
import logging
import torch
import math
import pickle
import utils_train
import utils_dist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import gaussian_kde

# Setup the argument parser
parser = argparse.ArgumentParser("Generate importance samples based on predictions by the NN")
parser.add_argument("--snr", type=str, default="10, 15, 20")
parser.add_argument("--num_samples", type=int, default=100)

parser.add_argument("--sample_sz_minimum", type=int, default=1000)
parser.add_argument("--sample_sz_maximum", type=int, default=50000)
parser.add_argument("--sample_sz_step", type=int, default=1000)
parser.add_argument("--num_repeats", type=int, default=10)

if __name__ == '__main__':
    # Given arguments
    args = parser.parse_args()

    # disable bilby
    logger = logging.getLogger('bilby')
    logger.disabled = True

    # Loop over the saved signals
    for snr in list(map(int, args.snr.split(","))):
        sample_sizes = np.arange(args.sample_sz_minimum, args.sample_sz_maximum + args.sample_sz_step, args.sample_sz_step)
        collection_logpdf = np.zeros((args.num_samples, len(sample_sizes)))
        is_good = np.zeros((args.num_samples,), dtype=np.bool)
        for sample_nb in tqdm(range(args.num_samples)):
            with open(f'samples/neural_data_{snr}_{sample_nb}.pck', 'rb') as f:
                neural_data = pickle.load(f)
                df = pd.read_csv(f'samples/imp_samples_{snr}_{sample_nb}.csv')

                # Extract useful info from the dataframe and pickled data
                tar = utils_train.decra2xyz(torch.from_numpy(neural_data['para']['dec']).unsqueeze(0),
                                            torch.from_numpy(neural_data['para']['ra']).unsqueeze(0))
                xyz = torch.Tensor(df[['x', 'y', 'z']].values)
                loggdiff = df['logdiff']

                if torch.acos((xyz[0, :] * torch.Tensor(neural_data['out_celestial'])).sum(axis=0, keepdims=True)) * 180 / math.pi < 90:
                    is_good[sample_nb] = 1

                for idx, sample_sz in enumerate(sample_sizes):
                    nb_succes = 0
                    for j in range(args.num_repeats):
                        # simulate a single run
                        indices = np.random.choice(xyz.shape[0], sample_sz, replace=False)
                        local_logdiff = loggdiff[indices]
                        local_logdiff = np.exp(local_logdiff - max(local_logdiff))

                        try:
                            kde_ra = gaussian_kde(df.loc[indices, 'ra'], weights=local_logdiff)
                            if kde_ra.logpdf(neural_data['para']['ra']) > 1:
                                collection_logpdf[sample_nb, idx] += kde_ra.logpdf(neural_data['para']['ra'])
                                nb_succes += 1
                        except:
                            pass
                    if nb_succes < 3:
                        is_good[sample_nb] = 0
                    else:
                        collection_logpdf[sample_nb, idx] /= (nb_succes + 1e-10)

        collection_logpdf = collection_logpdf[is_good]

        # print(collection_maae)
        plt.style.use('seaborn-pastel')
        plt.plot(sample_sizes, collection_logpdf.mean(axis=0), label=f'SNR={snr}')

    plt.legend()
    plt.ylabel('loglikelihood true $ra$')
    plt.xlabel('Importance samples')
    plt.xlim([0, args.sample_sz_maximum])
    plt.show()