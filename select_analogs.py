from scipy import spatial
import numpy as np
import pandas as pd

# Specifically written scripts for this work
from mw_analog_auto import mw_analog_auto


def select_analogs(galaxies, mw, n_analogs=5000, **kwargs):
    """Selects a single Milky Way analog from the given catalogs.
 
    Given a set of Milky Way parameters (e.g. M*, SFR, B2Tr, Rd, etc)
    will select a set of Milky Way analogs from a binary tree constructed
    from the parameters using the mw_analog_auto() function.

    It is assumed that the inputs are all in the same spacing. For example
    if the catalog used is log(M*), the mean and sigma of the Milky Way
    M* must also be log values.

    Args:
        galaxies (dataframe): A catalog of the parameters where analogs are
            being searched for. Each column denotes a parameter, each row
            denotes a galaxy.
            Example:
                galaxies = pd.DataFrame(
                    {"x": logmass_matched, 
                    "y": 10.**logsfr_matched}
                    )
        mw (dataframe): Contains the measured Milky Way values. Each column 
            denotes a parameter, the first row is the mean, and the second 
            row is the standard deviation of the parameter.
            Example:
                mstar = [mw_mean_log_mstar, mw_sigma_log_mstar]
                sfr = [mw_mean_sfr, mw_sigma_sfr]
                mw = pd.DataFrame(
                    {"x": mstar, "y": sfr}, 
                    index=["mean", "sigma"]
                    )
        n_analogs (int): Default is 5000. The number of analogs to obtain.

    Returns:
        An array of indices of a sample of MW-like galaxies in the 
        parameter space.

    """

    search_space = np.zeros((galaxies.shape[0], galaxies.shape[1]))
    for j, i in enumerate(galaxies.columns):
        search_space[:, j] = (galaxies[i].values - mw.at["mean", i]) / mw.at["sigma", i]
    tree = spatial.cKDTree(search_space)
    mwanalogs = []
    dists = []
    while np.size(mwanalogs) < n_analogs:
        mw_realization = mw.copy(deep=True)
        mw_realization.loc["point"] = [
            np.random.normal(loc=it[0], scale=it[1]) for it in (mw.T.values)
        ]
        mwa_index_i, dist_i = mw_analog_auto(tree, mw_realization, **kwargs)
        if np.isnan(mwa_index_i) == False:
            mwanalogs.append(mwa_index_i)
            dists.append(dist_i)
    mwanalogs = np.array(mwanalogs)
    dists = np.array(dists)
    return mwanalogs#, dists
