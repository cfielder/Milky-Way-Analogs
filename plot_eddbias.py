import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/NewDocuments/GitHub/MW_Morphology/')

arrays = Path.home() / "NewDocuments" / "GitHub" / "MW_Morphology" / "gp_calculations"
bias_means = np.load(arrays / "bias_means.npy")
bias_sigmas = np.load(arrays / "bias_sigmas.npy")
delta_mean_Mr = np.load(arrays / "delta_mean_Mr.npy")
delta_sigma_Mr = np.load(arrays / "delta_sigma_Mr.npy")
prediction_sets = np.load(arrays / "prediction_sets.npy")
std_sets = np.load(arrays / "std_sets.npy")

fig = plt.figure()
plt.hist(bias_means)
plt.xlabel(r"$M_{r}$ Bias Mean")
plt.show()
print(np.std(bias_means))

fig = plt.figure()
plt.hist(bias_sigmas)
plt.xlabel(r"$M_{r}$ Bias Sigma")
plt.show()

fig = plt.figure()
plt.hist(prediction_sets[:,0])
plt.xlabel(r"$M_{r}$ Non-biased Mean")
plt.show()
print(np.std(prediction_sets[:,0]))

fig = plt.figure()
plt.hist(std_sets[:,0])
plt.xlabel(r"$M_{r}$ Non-biased Std")
plt.show()

corr_prop = prediction_sets[:,0] - bias_means
corr_sigma = np.sqrt(std_sets[:,0] ** 2 + bias_sigmas ** 2)

fig = plt.figure()
plt.hist(corr_prop)
plt.xlabel(r"$M_{r}$ Corrected")
plt.show()
print(np.std(corr_prop))

fig = plt.figure()
plt.hist(corr_sigma)
plt.xlabel(r"$M_{r}$ Std Corrected")
plt.show()