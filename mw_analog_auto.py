import numpy as np
import sklearn
from sklearn import linear_model
import scipy
from scipy import spatial
import random

def mw_analog_auto(
    tree,
    mw,
    n_neighbors=7,
    max_sigma=6,
    silent=True,
    equal_weight=False,
    delta_cutoff=False,
    **kwargs
):
    """Selects a single Milky Way analog from the given catalogs.
 
    Given a Milky Way M*, SFR (or sSFR), B2Tr, or Rd will return the single 
    index value of a MW-like galaxy randomly selected from the 7 nearest 
    neighbors in a cKDTree of the given catalogs around a point drawn from 
    the Milky Way's posterior probability distribution. 

    It is assumed that the inputs are all in the same spacing. For example
    if the catalog used is log(M*), the mean and sigma of the Milky Way
    M* must also be log values.

    Args:
        tree (KDTree/binary tree): A tree created from catalogs being used to 
            find analogs in. Make sure these values are matched in RA and Dec, 
            and the data is scaled approiately. 
            Example:
                tree = spatial.cKDTree(search_space)
        mw (dataframe): A dataframe that contains the measured Milky Way values,                        and the random point drawn from the posterior probability 
            distribution.
            Example:
                mstar = [mw_mean_log_mstar, mw_sigma_log_mstar]
                sfr = [mw_mean_sfr, mw_sigma_sfr]
                for it in (mstar, sfr):
                    it.append(np.random.normal(loc=it[0], scale=it[1]))
                mw = pd.DataFrame({"x": mstar, "y": sfr}, 
                    index=["mean", "sigma", "point"])
        n_neighbors (int): Default is 7. The number of neighbors to search
            for around the given point.
        max_sigma (int): Default is 6. The maximum number of sigma used to 
            calculate the upper bound distance for the query. Typically between
            5 and 7.
        silent (bool): Default is True, which makes the function not return 
            print functions. If set to False the print statements are output.
        equal_weight (bool): Default is False, so the weighting in the selection 
            of the analogs from the n nearest neighbors depends on how far the
            neighbor is from the search point. If set to True then the 
            weighting is equal/the default of numpy.random.choice()

    Returns:
        An int representing the index of a single MW-like galaxy.

    Raises:
        ValueError: If there is only 1 parameter input. The minimum is 2.

    """
    n_params = int(len(list(mw)))
    if not silent:
        print("You have {} parameter input.".format(n_params))
    if n_params == 1:
        raise ValueError("You have too few parameters.")

    # Find nearest neighbors
    # First we must scale the data correctly
    base_point = np.zeros((n_params))
    columns = list(mw)
    for j, i in enumerate(columns):
        base_point[j] = (mw.at["point", i] - mw.at["mean", i]) / mw.at["sigma", i]
    # Calculate upper bound distance
    upper_bound = np.sqrt(np.sum((mw.loc['mean',:]+max_sigma*mw.loc['sigma',:])**2))
    # Find the n_neighbors nearest neightbors
    dist, inds = tree.query(base_point, k=n_neighbors, distance_upper_bound=max_sigma)
    # Randomly select one of the neighbors
    if n_neighbors > 1:
        if equal_weight is False:
            weight = (np.exp(-dist ** 2 / 2)) / np.sum(np.exp(-dist ** 2 / 2))
        else:
            weight = None
        analog = np.random.choice(inds, p=weight)
        wh_dist = np.where(inds == analog)
        dist = dist[wh_dist]
    else:
        analog = inds
    # (Optional) Restrict analog to be within 0.5 sigma or drop
    if delta_cutoff:
        if dist > 0.5:
            analog = np.nan

    return analog, dist
