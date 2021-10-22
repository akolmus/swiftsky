import pandas as pd
import healpy as hp
import numpy as np
import ligo.skymap
import argparse
import matplotlib.pyplot as plt

from matplotlib import rcParams
from pathlib import Path
from ligo.skymap import io, kde, postprocess
from ligo.skymap.plot.marker import reticle


# Setup the argument parser
parser = argparse.ArgumentParser("Generate importance samples based on predictions by the NN")
parser.add_argument("--name", type=str, default="with_sky_values")
parser.add_argument("--snr", type=str, default="20")
parser.add_argument("--num_events", type=int, default=10)
parser.add_argument("--num_samples", type=int, default=2000)
parser.add_argument("--weighted", type=bool, default=True)
parser.add_argument("--trials", type=int, default=5)
parser.add_argument("--show_plot", type=bool, default=False)


def generate_fits(samples: pd.Series, trials: int, label: str):
    """ """

    # Create kernel density estimation
    pts = samples[['ra', 'dec']].values
    sky_posterior = kde.Clustered2DSkyKDE(pts, trials=trials, jobs=1)

    # Convert to healpix map and save as fits file
    hpmap = sky_posterior.as_healpix()
    io.write_sky_map(f'skymaps/{label}.fits', hpmap, nest=True)


def fits2skymap(label: str, show: bool = True, path: str = None, contour: list = [90]):
    """ """
    # Load skymap
    if not path:
        skymap, metadata = io.fits.read_sky_map(f'skymaps/{label}.fits', nest=None)
        nside = hp.npix2nside(len(skymap))
    else:
        skymap, metadata = io.fits.read_sky_map(path, nest=None)
        nside = hp.npix2nside(len(skymap))

    # Convert to probability per square degree
    deg2perpix = hp.nside2pixarea(nside, degrees=True)
    probperdeg2 = skymap / deg2perpix

    ax = plt.axes(projection="astro hours mollweide")
    ax.grid()

    vmax = probperdeg2.max()
    ax.imshow_hpx((probperdeg2, 'ICRS'), nested=True, vmin=0, vmax=vmax, cmap='Purples')
    if contour is not None:
        confidence_levels = 100 * postprocess.find_greedy_credible_levels(skymap)
        contours = ax.contour_hpx((confidence_levels, 'ICRS'), nested=metadata['nest'], colors='k', linewidths=0.5, levels=contour)

    plt.savefig(f'skymaps/{label}.png', dpi=600)
    if show:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # Given arguments
    args = parser.parse_args()

    # Loop over the saved signals
    for snr in list(map(int, args.snr.split(","))):
        for event_nb in range(args.num_events):
            label = f'VMF_{snr}_{event_nb}' if not args.weighted else f'IMP_{snr}_{event_nb}'

            imp_samples = pd.read_csv(f'samples/imp_samples_{args.name}_{snr}_{event_nb}.csv')
            imp_weights = np.exp(imp_samples['logdiff'] - max(imp_samples['logdiff']))
            imp_samples = imp_samples.sample(n=args.num_samples) if not args.weighted else imp_samples.sample(n=args.num_samples, weights=imp_weights, replace=True)

            if not Path(f'skymaps/{args.name}_{label}.fits').exists():
                generate_fits(imp_samples, trials=args.trials, label=f"{args.name}_{label}")
            fits2skymap(label=f"{args.name}_{label}", show=args.show_plot)
