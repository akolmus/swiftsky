import argparse
import warnings
import logging
import torch
import bilby
import pickle
import utils_train
import utils_dist
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from data.gwave import build_prior
from bilby.gw import WaveformGenerator
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.detector import InterferometerList, Interferometer

# Setup the argument parser
parser = argparse.ArgumentParser("Generate importance samples based on predictions by the NN")
parser.add_argument("--name", type=str, default="with_sky_values")
parser.add_argument("--snr", type=str, default="10, 15, 20")
parser.add_argument("--num_samples", type=int, default=20)
parser.add_argument("--num_imp_samples", type=int, default=100000)

# Data arguments
parser.add_argument("--H1_active", type=bool, default=True)
parser.add_argument("--L1_active", type=bool, default=True)
parser.add_argument("--V1_active", type=bool, default=True)
parser.add_argument("--sampling_frequency", type=int, default=2048)
parser.add_argument("--duration", type=float, default=2.0)
parser.add_argument("--merger_position", type=float, default=0.5)
parser.add_argument("--waveform_approximant", type=str, default="IMRPhenomPv2", choices=["IMRPhenomPv2"])
parser.add_argument("--reference_frequency", type=float, default=50.0)
parser.add_argument("--minimum_frequency", type=float, default=20.0)

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
parser.add_argument("--geocent_time", type=bool, default=True)

# Sampling arguments
parser.add_argument("--set_to_true_value", type=str, default="luminosity_distance, chirp_mass, mass_ratio, theta_jn, psi, phase, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, chi_1, chi_2, geocent_time")


def sample_func(args, vmf, mvg, prior, req_samples):
    """ Get samples, by combining the vmf, mvg and prior, pass along xyz for easy handling later """
    prior_samples = prior.sample(req_samples)

    # Sample the chirp mass and mass_ratio from the mvg
    if 'chirp_mass' not in args.set_to_true_value:
        masses = mvg.sample((10 * req_samples,)).squeeze(1)
        masses, _ = torch.sort(masses, dim=1)
        masses = masses[masses[:, 0] > args.mass_minimum]
        masses = masses[masses[:, 1] < args.mass_maximum]
        masses = masses[:req_samples]
        prior_samples['mass_1'] = masses[:, 1].numpy()
        prior_samples['mass_2'] = masses[:, 0].numpy()
        prior_samples['mass_ratio'] = bilby.gw.conversion.component_masses_to_mass_ratio(masses[:, 1], masses[:, 0]).numpy()
        prior_samples['chirp_mass'] = bilby.gw.conversion.component_masses_to_chirp_mass(masses[:, 1], masses[:, 0]).numpy()
    else:
        total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(prior_samples['chirp_mass'], prior_samples['mass_ratio'])
        prior_samples['mass_1'], prior_samples['mass_2'] = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(prior_samples['mass_ratio'], total_mass)

    # Sample the vmf for the sky coordinates
    xyz = vmf.sample(req_samples)
    prior_samples['dec'], prior_samples['ra'] = utils_train.xyz2decra(xyz)
    prior_samples['dec'], prior_samples['ra'] = prior_samples['dec'].numpy(), prior_samples['ra'].numpy()
    prior_samples['x'] = xyz[:, 0].numpy()
    prior_samples['y'] = xyz[:, 1].numpy()
    prior_samples['z'] = xyz[:, 2].numpy()

    # Convert to pandas dataframe
    return pd.DataFrame().from_dict(prior_samples), xyz


def importance_sampling(args, wave, vmf, mvg, para, req_samples):
    """ """
    # Setup injection parameters
    injection_parameters = {key: float(value) for key, value in para.items()}

    # Setup waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(duration=args.duration,
                                                    sampling_frequency=args.sampling_frequency,
                                                    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                    parameters=injection_parameters,
                                                    waveform_arguments={'waveform_approximant': args.waveform_approximant,
                                                                        'reference_frequency': args.reference_frequency,
                                                                        'minimum_frequency': args.minimum_frequency})

    # Setup interferometers
    ifos = InterferometerList(["H1"] * args.H1_active + ["L1"] * args.L1_active + ["V1"] * args.V1_active)
    ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=args.sampling_frequency,
                                                       duration=args.duration,
                                                       start_time=1e9 - args.duration + args.merger_position)
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

    # Obtain the correct prior
    priors = build_prior(args)
    for key, value in injection_parameters.items():
        if key in args.set_to_true_value.split(', '):
            priors[key] = value

    # Obtain likelihood calculator and setup sampler
    likelihood = bilby.gw.GravitationalWaveTransient(interferometers=ifos, waveform_generator=waveform_generator)
    priors.fill_priors(likelihood=likelihood)
    sampler = bilby.core.sampler.dynesty.Dynesty(likelihood, priors)

    # Obtain samples
    prior_samples, cart_coord = sample_func(args, vmf, mvg, priors, req_samples)
    logdiff = np.zeros((req_samples,))
    for idx, sample in prior_samples.iterrows():
        parsed_sample = [sample[key] for key in injection_parameters.keys() if key not in ['mass_1', 'mass_2', 'luminosity_distance', 'x', 'y', 'z']]
        logdiff[idx] = sampler.log_likelihood(parsed_sample) - (vmf.logpdf(cart_coord[idx:idx+1] + mvg.log_prob(torch.Tensor([sample['mass_1'], sample['mass_2']]))))

    # Correct for false flags
    prior_samples['logdiff'] = logdiff
    prior_samples = prior_samples[prior_samples['logdiff'] > -10 ** 5]

    return prior_samples


if __name__ == '__main__':
    # Given arguments
    args = parser.parse_args()

    # disable bilby
    logger = logging.getLogger('bilby')
    logger.disabled = True

    # Loop over the saved signals
    for snr in list(map(int, args.snr.split(","))):
        maae = torch.zeros((args.num_samples))
        total_kappa = 0
        for sample_nb in tqdm(range(args.num_samples)):
            with open(f'samples/neural_data_{args.name}_{snr}_{sample_nb}.pck', 'rb') as f:
                neural_data = pickle.load(f)

            if Path(f'samples/imp_samples_{args.name}_{snr}_{sample_nb}.csv').exists():
                df = pd.read_csv(f'samples/imp_samples_{args.name}_{snr}_{sample_nb}.csv')
                num_imp_sampled = df.shape[0]
            else:
                df = pd.DataFrame(columns=["mass_ratio", "chirp_mass", "luminosity_distance", "dec", "ra", "theta_jn", "psi", "phase",
                                           "chi_1", "chi_2", "geocent_time", "mass_1", "mass_2", "x", "y", "z", "logdiff"])
                num_imp_sampled = 0

            if num_imp_sampled < args.num_imp_samples:
                para = neural_data['para']
                kappa = neural_data['kappa']
                out_celestial = neural_data['out_celestial']
                out_mass_mean = neural_data['out_mass_mean']
                out_mass_cov = neural_data['out_mass_cov']

                # Setup distributions
                out_mass_mean, out_mass_cov = utils_train.output2mu_sigma(torch.cat([out_mass_mean.unsqueeze(0), out_mass_cov.unsqueeze(0)], dim=1))
                vmf = utils_dist.VMFDistribution(kappa=kappa, mu=out_celestial.squeeze())
                mvg = torch.distributions.MultivariateNormal(loc=50 * out_mass_mean + 30, covariance_matrix=2500 * out_mass_cov)

                # Create the samples
                samples = importance_sampling(args, 0, vmf, mvg, para, int(1.01*(args.num_imp_samples - num_imp_sampled)))
                df = df.append(samples)
                df = df.iloc[:args.num_imp_samples]
                df.to_csv(f'samples/imp_samples_{args.name}_{snr}_{sample_nb}.csv', index=False)