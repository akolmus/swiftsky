import time

import numpy as np
import bilby
import torch

from torch.utils.data import Dataset
from scipy.interpolate import interp1d

from bilby.gw import WaveformGenerator
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.detector import InterferometerList
from bilby.gw.prior import AlignedSpin, BBHPriorDict
from bilby.core.utils import infft, create_frequency_series
from bilby.core.prior import Constraint, Cosine, Sine, Uniform


def build_prior(args) -> BBHPriorDict:
    """ Apply the specified settings to the prior """

    # Check some logic
    assert args.chirp_mass_maximum >= args.chirp_mass_minimum, ""
    assert args.mass_ratio_maximum >= args.mass_ratio_minimum, ""
    assert args.mass_maximum >= args.mass_minimum, ""
    assert args.spin_maximum >= args.spin_minimum, ""

    # Obtain prior
    prior = bilby.gw.prior.BBHPriorDict(aligned_spin=args.aligned_spin)

    # Fill in the mass parameters
    prior['mass_1'] = Constraint(minimum=args.mass_minimum, maximum=args.mass_maximum, name='mass_1', latex_label='$m_1$', unit=None)
    prior['mass_2'] = Constraint(minimum=args.mass_minimum, maximum=args.mass_maximum, name='mass_2', latex_label='$m_1$', unit=None)
    prior['mass_ratio'] = Uniform(minimum=args.mass_ratio_minimum, maximum=args.mass_ratio_maximum, name='mass_ratio', latex_label='$q$', unit=None, boundary=None)
    prior['chirp_mass'] = Uniform(minimum=args.chirp_mass_minimum, maximum=args.chirp_mass_maximum, name='chirp_mass', latex_label='$\\mathcal{M}$', unit=None, boundary=None)

    # The luminosity distance, since we change the SNR the luminosity distance is set to be a constant
    prior['luminosity_distance'] = 1000.0

    # The celestial angles
    prior['dec'] = Cosine(minimum=-1.5707963267948966, maximum=1.5707963267948966, name='dec', latex_label='$\\mathrm{DEC}$', unit=None,  boundary=None)
    prior['ra'] = Uniform(minimum=0, maximum=6.283185307179586, name='ra', latex_label='$\\mathrm{RA}$', unit=None, boundary='periodic')

    # Some angle TODO: what is the physical interpretation of this thing
    prior['theta_jn'] = Sine(name='theta_jn', latex_label='$\\theta_{JN}$', unit=None, minimum=0, maximum=3.141592653589793, boundary=None)

    # Some angle TODO: what is the physical interpretation of this thing
    prior['psi'] = Uniform(minimum=0, maximum=3.141592653589793, name='psi', latex_label='$\\psi$', unit=None, boundary='periodic')

    # Phase
    prior['phase'] = Uniform(minimum=0, maximum=6.283185307179586, name='phase', latex_label='$\\phi$', unit=None, boundary='periodic')

    # The spin parameters
    if args.aligned_spin:
        prior['chi_1'] = AlignedSpin(a_prior=Uniform(minimum=args.spin_minimum, maximum=args.spin_maximum, name=None, latex_label=None, unit=None, boundary=None), z_prior=Uniform(minimum=-1, maximum=1, name=None, latex_label=None, unit=None, boundary=None), name='chi_1', latex_label='$\\chi_1$', unit=None, boundary=None, minimum=-0.99, maximum=0.99)
        prior['chi_2'] = AlignedSpin(a_prior=Uniform(minimum=args.spin_minimum, maximum=args.spin_maximum, name=None, latex_label=None, unit=None, boundary=None), z_prior=Uniform(minimum=-1, maximum=1, name=None, latex_label=None, unit=None, boundary=None), name='chi_2',latex_label='$\\chi_2$', unit=None, boundary=None, minimum=-0.99, maximum=0.99)
    else:
        prior['a_1'] = Uniform(minimum=args.spin_minimum, maximum=args.spin_maximum, name='a_1', latex_label='$a_1$', unit=None, boundary=None) if args.spin_maximum > args.spin_minimum else 0
        prior['a_2'] = Uniform(minimum=args.spin_minimum, maximum=args.spin_maximum, name='a_2', latex_label='$a_2$', unit=None, boundary=None) if args.spin_maximum > args.spin_minimum else 0
        prior['tilt_1'] = Sine(name='tilt_1', latex_label='$\\theta_1$', unit=None, minimum=0, maximum=3.141592653589793, boundary=None) if args.tilt1 else 0
        prior['tilt_2'] = Sine(name='tilt_2', latex_label='$\\theta_2$', unit=None, minimum=0, maximum=3.141592653589793, boundary=None) if args.tilt2 else 0
        prior['phi_12'] = Uniform(minimum=0, maximum=6.283185307179586, name='phi_12', latex_label='$\\Delta\\phi$', unit=None, boundary='periodic') if args.phi_12 else 0
        prior['phi_jl'] = Uniform(minimum=0, maximum=6.283185307179586, name='phi_jl', latex_label='$\\phi_{JL}$', unit=None, boundary='periodic') if args.phi_jl else 0

    # Add the arrival time
    prior['geocent_time'] = Uniform(1e9 - 0.1, maximum=1e9 + 0.1, name='geocent_time', latex_label='$t_c$', unit='$s$') if args.geocent_time else 1e9

    return prior


class GravitationalWaveDataset(Dataset):
    """ """

    def __init__(self, args, train: bool):
        """ """

        # Setup
        self.args = args
        self.num_samples = self.args.trn_samples if train else self.args.val_samples

        # Set SNR values
        self.snr_min = args.snr_min
        self.snr_max = args.snr_max
        self.snr_peak = args.snr_peak
        self.snr_temp = args.snr_temp
        self.c1 = 1 + self.snr_temp * (self.snr_peak - self.snr_min) / (self.snr_max - self.snr_min)
        self.c2 = 1 + self.snr_temp * (self.snr_max - self.snr_peak) / (self.snr_max - self.snr_min)

        # Create correct frequency series
        self.freq_series = create_frequency_series(args.sampling_frequency, args.initial_duration)

        # Create waveform generator
        self.waveform_generator = WaveformGenerator(duration=args.initial_duration,
                                                    sampling_frequency=args.sampling_frequency,
                                                    frequency_domain_source_model=lal_binary_black_hole,
                                                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                    waveform_arguments={'waveform_approximant': args.waveform_approximant,
                                                                        'reference_frequency': args.reference_frequency,
                                                                        'minimum_frequency': args.minimum_frequency})

        # Setup interferometers
        self.detectors = ["H1"] * args.H1_active + ["L1"] * args.L1_active + ["V1"] * args.V1_active
        self.interferometers = InterferometerList(self.detectors)
        self.interferometers.set_strain_data_from_zero_noise(args.sampling_frequency, args.initial_duration, 1e9 - args.initial_duration + args.merger_position)

        # Obtain the psd and asd noise curves
        self.psd, self.asd = self.load_frequency_noise_curves()

        # Obtain prior
        self.prior = build_prior(args)

    def load_frequency_noise_curves(self):
        """ Return the power and amplitude spectral density curves from H1, L1, V1 """

        # Load the psd / asd from file
        hl_freq_mask, hl_asd = np.genfromtxt("data/ncurves/aLIGO_O4_high_asd.txt").T
        hl_psd = hl_asd ** 2
        v_freq_mask, v_psd = np.genfromtxt("data/ncurves/AdV_psd.txt").T
        v_asd = v_psd ** 0.5

        # Setup the interpolation for psd and asd
        psd_interpolated = {'H1': interp1d(hl_freq_mask, hl_psd, bounds_error=False, fill_value=np.inf),
                            'L1': interp1d(hl_freq_mask, hl_psd, bounds_error=False, fill_value=np.inf),
                            'V1': interp1d(v_freq_mask, v_psd, bounds_error=False, fill_value=np.inf)}

        asd_interpolated = {'H1': interp1d(hl_freq_mask, hl_asd, bounds_error=False, fill_value=np.inf),
                            'L1': interp1d(hl_freq_mask, hl_asd, bounds_error=False, fill_value=np.inf),
                            'V1': interp1d(v_freq_mask, v_asd, bounds_error=False, fill_value=np.inf)}

        # Setup the curves
        asd = np.zeros(shape=(len(self.detectors), len(self.freq_series)))
        psd = np.zeros(shape=(len(self.detectors), len(self.freq_series)))

        # Setup the psd and asd
        for i, detector in enumerate(self.detectors):
            psd[i] = psd_interpolated[detector](self.freq_series)
            asd[i] = asd_interpolated[detector](self.freq_series)

        return psd, asd

    def set_snr_values(self, snr_min: float, snr_max: float, snr_peak: float, snr_temp: float):
        """ Set the settings for the distribution to draw SNR values from """
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.snr_peak = snr_peak
        self.snr_temp = snr_temp
        self.c1 = 1 + self.snr_temp * (self.snr_peak - self.snr_min) / (self.snr_max - self.snr_min + 1e-8)
        self.c2 = 1 + self.snr_temp * (self.snr_max - self.snr_peak) / (self.snr_max - self.snr_min + 1e-8)

    def generate_snr_value(self):
        """ Generate a value from a stretched Beta distribution, also known as  """
        if self.snr_max != self.snr_min:
            return self.snr_min + (self.snr_max - self.snr_min) * np.random.beta(self.c1, self.c2)
        else:
            return self.snr_max

    def generate_white_noise(self):
        """ Based on the Bilby create white noise function """

        # Set random (white) frequency parts
        norm1 = 0.5 * self.args.initial_duration ** 0.5
        re1 = np.random.normal(0, norm1, size=(len(self.detectors), len(self.freq_series)))
        im1 = np.random.normal(0, norm1, size=(len(self.detectors), len(self.freq_series)))

        # convolve data with instrument transfer function
        wnoise = (re1 + 1j * im1) * 1.

        # set DC and Nyquist = 0
        wnoise[:, 0] = 0

        # no Nyquist frequency when N=odd
        if np.mod(int(np.round(self.args.initial_duration * self.args.sampling_frequency)), 2) == 0:
            wnoise[:, -1] = 0

        return wnoise

    def batch_noise(self, num_samples: int):
        """ Generate noise strains for the normalization process """

        noise_strain = torch.zeros(size=(num_samples, len(self.detectors), int(self.args.cut_duration * self.args.sampling_frequency)))

        for i in range(num_samples):
            # Generate White noise
            white_noise = self.generate_white_noise()
            noise = self.psd ** 0.5 * white_noise
            noise[:, self.freq_series < self.args.minimum_frequency] = 0 * (1 + 1j)
            noise[:, self.freq_series > self.args.maximum_frequency] = 0 * (1 + 1j)

            # Whiten the signal data and add the noise
            freq_strain = noise / self.asd if self.args.whiten_time_series else noise

            # Obtain time strain
            time_strain = np.float32(infft(freq_strain, self.args.sampling_frequency))

            # Cut the timestrain to the rightsize
            time_strain = time_strain[:, int((time_strain.shape[1] - self.args.cut_duration * self.args.sampling_frequency) / 2):int((time_strain.shape[1] + self.args.cut_duration * self.args.sampling_frequency) / 2)]

            # Add to the noise strain
            noise_strain[i] = torch.Tensor(time_strain)

        return noise_strain

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get parameters
        parameters = self.prior.sample(size=1)
        parameters = {key: item[0] for key, item in parameters.items()}

        # Convert to the Bilby format
        polarizations = self.waveform_generator.frequency_domain_strain(parameters)

        # Setup the frequency strain
        freq_strain = np.zeros((len(self.detectors), self.freq_series.shape[0]), dtype=np.complex_)
        for i in range(len(self.detectors)):
            freq_strain[i] = self.interferometers[i].get_detector_response(polarizations, parameters)

        # Calculate the optimal SNR for each detector and subsequently the scalefactor for the desired SNR
        snr = np.real((4 / self.args.initial_duration * np.sum(np.conj(freq_strain) * freq_strain / self.psd, axis=1)) ** 0.5)
        scale_factor = self.generate_snr_value() / np.sqrt(np.sum(snr ** 2))

        # Rescale the luminosity distance such that it matches the rescaled signal
        parameters['luminosity_distance'] = parameters['luminosity_distance'] / scale_factor

        # Generate White noise
        white_noise = self.generate_white_noise()
        noise = self.psd ** 0.5 * white_noise
        noise[:, self.freq_series < self.args.minimum_frequency] = 0 * (1 + 1j)
        noise[:, self.freq_series > self.args.maximum_frequency] = 0 * (1 + 1j)

        # Whiten the signal data and add the noise
        freq_strain = scale_factor * freq_strain + self.args.inject_noise * noise
        freq_strain = freq_strain / self.asd if self.args.whiten_time_series else freq_strain
        freq_strain[:, self.freq_series < self.args.minimum_frequency] = 0 * (1 + 1j)
        freq_strain[:, self.freq_series > self.args.maximum_frequency] = 0 * (1 + 1j)

        # Obtain time strain
        time_strain = np.float32(infft(freq_strain, self.args.sampling_frequency))

        # Cut the timestrain to the rightsize
        time_strain = time_strain[:, time_strain.shape[1] - int(self.args.cut_duration * self.args.sampling_frequency):time_strain.shape[1]]

        return time_strain, parameters
