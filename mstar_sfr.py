import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from MW_params import *
from do_full_analysis import do_full_analysis
#from select_analogs import select_analogs

"""The purpose of this script is to select analogs and then do the various 
   photometry calculations on said analogs by accessing do_full_analysis.py.
"""

# Read in catalogs, update to your own directories
catalogs = Path.home() / "Catalogs"
cross_matched_catalog = pd.read_pickle(catalogs / "SDSS_matched_catalog")
print(cross_matched_catalog.columns)
np.save("sample_Mr",cross_matched_catalog.cmodel_M_r)
prop = np.load(catalogs / "prop_sdss.npy")
# Create dataframes that are needed for passing into various functions
galaxies = cross_matched_catalog[["logmass", "sfr"]].copy(deep=True)
galaxies.rename(columns={"logmass": "M*"}, inplace=True)
galaxies.rename(columns={"sfr": "SFR"}, inplace=True)
galaxies_sigma = cross_matched_catalog[["sigma_logmass", "sigma_sfr"]].copy(deep=True)
galaxies_sigma.rename(columns={"sigma_logmass": "M*"}, inplace=True)
galaxies_sigma.rename(columns={"sigma_sfr": "SFR"}, inplace=True)
mw_mstar = [mw_mean_log_mstar, mw_sigma_log_mstar]
#mw_mstar = [old_mean_mstar,old_sigma_mstar]
mw_sfr = [mw_mean_sfr, mw_sigma_sfr]
mw = pd.DataFrame({"M*": mw_mstar, "SFR": mw_sfr}, index=["mean", "sigma"])

do_full_analysis(
    galaxies,
    galaxies_sigma,
    mw,
    cross_matched_catalog,
    prop,
    filename = "mstar_sfr_analogs.npy",
    get_analogs=True,
    get_plots=True,
    get_photo_analysis=True,
    delta_cutoff=False
)

#TESTING#
#analogs,dists = (select_analogs(galaxies,mw,n_analogs=5000,n_neighbors=7,delta_cutoff=False))
#plt.hist(dists)
#plt.xlabel("Neighbor distance")
#plt.title("M*-SFR")
#plt.show()
