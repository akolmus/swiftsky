import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta

snr_min = 10
snr_max = 50
snr_peak = 15
snr_temp = 15
c1 = 1 + snr_temp * (snr_peak - snr_min) / (snr_max - snr_min + 1e-8)
c2 = 1 + snr_temp * (snr_max - snr_peak) / (snr_max - snr_min + 1e-8)

beta_dist = beta(c1, c2)
snr_pdf = beta_dist.pdf(np.arange(0, 1.001, 0.001))

plt.style.use('seaborn-pastel')
plt.fill_between(snr_min + (snr_max - snr_min) * np.arange(0, 1.001, 0.001), snr_pdf)
plt.xlim([10, 50])
plt.ylim(ymin=0)
plt.ylabel('pdf')
plt.xlabel('Optimal SNR')
plt.show()