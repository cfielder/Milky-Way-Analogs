########################################################################################
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel
import scipy
from scipy import interpolate, integrate, spatial
from sklearn import linear_model
from matplotlib import pyplot as plt
import random
import time
import sys
import math
from pathlib import Path

# Specifically written scripts for this work
from mw_analog_auto import mw_analog_auto

########################################################################################
## USEFUL FUNCTIONS SECTION
########################################################################################
def get_index_bulge_disk(catalog, analog_indices):
    """Gives indices and sized of bulge vs. disk and edge on vs. face on

    Args:
        catalog (fits file): Catalog used to find these cuts, typically
            from your master cross-matched catalog.
        analog_indices (array): An array of indices of MW analogs selected
            from mw_analog_auto()

    Returns:
        wh_good (index array): which analogs are bulge dominated and/or face on
        n_good (int): how many of the sample is bulge dominated and/or face on
        wh_disk (index array): which analogs are disk dominated
        n_disk (int): how many of the sample is disk dominates
        wh_bulge (index array): which analogs are bulge dominated
        n_bulge (int): how many of the sample is bulge dominated
        disk_ratio: ratio of bulge dominated and elliptical galaxies to
            disk dominated galaxies
        f_disk: fraction of galaxies that are disk dominated in the sample  
    """
    is_disk_dom = catalog[analog_indices]["FRACPSF"] <= 0.5
    is_bulge_dom = catalog[analog_indices]["FRACPSF"] > 0.5
    is_edge_on = catalog[analog_indices]["AB_EXP"] <= 0.6
    is_face_on = catalog[analog_indices]["AB_EXP"] > 0.6

    wh_good = np.where(np.logical_or(is_bulge_dom, is_face_on))[0]
    n_good = np.size(wh_good)
    wh_disk = np.where(catalog[analog_indices[wh_good]]["FRACPSF"] <= 0.5)[0]
    n_disk = np.size(wh_disk)
    wh_bulge = np.where(catalog[analog_indices[wh_good]]["FRACPSF"] > 0.5)[0]
    n_bulge = np.size(wh_bulge)
    disk_ratio = np.true_divide(np.sum(is_disk_dom), n_disk)
    f_disk = np.true_divide(np.sum(is_disk_dom), np.size(analog_indices))

    return wh_good, n_good, wh_disk, n_disk, wh_bulge, n_bulge, disk_ratio, f_disk


########################################################################################
def hlmean(data, nsamp=None):
    """ Determines the H-L mean of a given data set

    Args:
        data (array): The values to calculate the H-L mean of.
        nsamp (optional; int): If set the H-L mean will use this number of bootstrap 
            samples to do the calculation. If not set it will default to 10 * the 
            number of elements in the data array.

    Returns:
        A float representing the bootstrap estimate of the H-L mean estimator.
    """
    ndata = len(data)
    if nsamp is None:
        nsamp = 10 * ndata
    data_boot = np.random.choice(data, size=(nsamp, 2))
    means = np.sum(data_boot, axis=1) / 2.0
    return np.median(means)


########################################################################################
def bootstrap(data, nboot=None):
    """ Creates a boostrapped sample of a given data set. Equal probability is assumed.

    Args:
        data (array): The values to bootstrap with replacement.
        nboot (optional; int): If set the data will be bootsrapped this number of times. 
            If not set it will default to 100.

    Returns:
        A 2D array of the bootstrapped sample in the shape of data length x resample size.
    """
    ndata = np.size(data)
    if nboot is None:
        nboot = 100
    data_resample = np.random.choice(data, size=(ndata, nboot))
    return data_resample


########################################################################################
def deriv(x, y):
    """ Calculates the numerical 3 point derivative of two given arrays.

    Args:
        x (array): The array to calculate the derivative of.
        y (array): The array to calculate the derivative with respect to.

    Returns:
        Floats of the derivative with size equal to the size of the input arrays. 
    """
    n = len(x)
    n2 = n - 2

    x12 = x - np.roll(x, -1)  # x1 - x2
    x01 = np.roll(x, 1) - x  # x0 - x1
    x02 = np.roll(x, 1) - np.roll(x, -1)  # x0 - x2

    # Middle points
    d = (
        np.roll(y, 1) * (x12 / (x01 * x02))
        + y * (1.0 / x12 - 1.0 / x01)
        - np.roll(y, -1) * (x01 / (x02 * x12))
    )

    # Formulae for the first and last points:
    d[0] = (
        y[0] * (x01[1] + x02[1]) / (x01[1] * x02[1])
        - y[1] * x02[1] / (x01[1] * x12[1])
        + y[2] * x01[1] / (x02[1] * x12[1])
    )
    d[n - 1] = (
        -y[n - 3] * x12[n2] / (x01[n2] * x02[n2])
        + y[n - 2] * x02[n2] / (x01[n2] * x12[n2])
        - y[n - 1] * (x02[n2] + x12[n2]) / (x02[n2] * x12[n2])
    )

    return d


########################################################################################
######################################
# Main Function #######################
######################################
########################################################################################
def mw_color_lum_stats(
    galaxies,
    galaxies_sigma,
    mw,
    mwanalogs,
    matched_catalog,
    prop,
    n_bins=10000,
    n_iters=100,
    n_mw=5000,
    n_hlmean_boots=100,
    n_boots=1000,
    raw_props=False,
    eddbias=False,
    corrected_props=False,
    derivatives=False,
    axistest=False,
):
    """Used for constructing useful SDSS MW analog galaxy samples with systematic errors
        taken into account. This is to be used after the sample has been cleaned and an 
        analog sample has been drawn.

    If all options are set to False then the function does not return anything.

    Args:
        galaxies (dataframe): A dataframe consisting of catalogs being used to 
            find analogs in. Make sure these values are matched in RA and Dec.
            Example:
                galaxies = pd.DataFrame({"x": logmass, "y": sfr})
        galaxies_sigma (dataframe): Same as galaxies but the errors of the measurements.
        mw (dataframe): A dataframe that contains the measured Milky Way values.
            Example:
                mstar = [mw_mean_log_mstar, mw_sigma_log_mstar]
                sfr = [mw_mean_sfr, mw_sigma_sfr]
                mw = pd.DataFrame({"x": mstar, "y": sfr}, 
                    index=["mean", "sigma"])
        mwanalogs (array): A 1D array of indices of Milky Way analogs in the desired
            parameter space. This is created using mw_analog_auto()
        matched_catalog (pandas dataframe): The main catalog (included cross-matching restriction and 
            volume limited sample cuts already) that the analogs are found in. This is generated
            in cross_matched_catalogs.py 
        prop (2d array): A 2D array containing the mag, color, and mass-to-light ratio of the 
            cross-matched galaxy sample.
        n_bins (int): Binning used in determining weights. Used in the raw_props section.
        n_iters (int): Number of realizations of perturbing MW analogs by a Gaussian
            in order to determine the Eddington bias. Used in the eddbias section.
        n_mw (int): The number of Milky Way analog galaxies being studied (same size
            mwanalogs).
        n_hlmean_boots (int): The number of bootstraps used when taking the hlmean. Used in
            the eddbias, derivatives, and axistest section. Refer to function for defaults.
        n_boots (int): The number of bootstraps used when testing cuts for inclination
            reddening. Used in the axistest section. Refer to function for defaults.
        raw_props (bool): If set to True this calculates the raw estimate for each property x.
            Determines the histogram of values reweighted to correct for being edge on.
            Calcaulates the CDF and measures x_50 +(x_84-x_50), -(x_50-x_16) where x_z 
                is the z-percentile.
        eddbias (bool): If set to True estimates the Eddington bias affecting each property x
            Perturbs the catalog of M*/SFR/etc values by random Gaussian noise characteristic
                of measurement errors.
            Does this iteratively to see how x_50 changes after 4 noise levels, fits a 
                quadrativ curve to delta_x_50 vs N_perturb to extrapolate the N=0 value.
            Requires raw_props to have been calculated.
        corrected_props (bool): If set to True calculates the corrected property estimates by 
            subtracting off the Eddington bias. Requires raw_props and eddbias to have
            been calculated.
        derivatives (bool): If set to True calculates derivatives of photometric properties. 
            It is assumed Eddington bias does not affect these.
        axistest (bool): If set to True tests different rejections of disks/spheriodal 
            galaxies by calculating mean props in 3 ways:
                0 - Rejecting all galaxies by inclination
                1 - Rejecting only disk galaxies by inclination
                2 - Same as 1, but calc. the weighted mean with disks weighted by their 
                    fraction before cut

    Returns:
        npz files of results from each section.
    """
    # Define where .npz files are saved
    # CHANGE TO MATCH WITH YOUR FILE STRUCTURE
    path_outputs = Path.home()

    # Cast matched_catalog from dataframe to astropy table for ease of use
    matched_catalog = Table.from_pandas(matched_catalog)

    if raw_props:
        # Make concise indexing array for analogs from original set and calculate weights
        # for mean properties
        wh_good, n_good, wh_disk, n_disk, wh_bulge, n_bulge, disk_ratio, f_disk = get_index_bulge_disk(
            matched_catalog, mwanalogs
        )

        # using mw analog indices calculate properties and their errors at z=0,0.1
        # set up empty arrays for storing mean values and percentiles
        mean_prop = np.empty((3, 2, 10))
        lse_prop = np.empty((3, 2, 10))
        hse_prop = np.empty((3, 2, 10))

        for p in range(3):
            for z in range(2):
                for bc in range(10):
                    mean_prop[p, z, bc] = (
                        n_disk
                        * disk_ratio
                        * hlmean(prop[p, z, bc, mwanalogs[wh_good][wh_disk]])
                        + n_bulge * hlmean(prop[p, z, bc, mwanalogs[wh_good][wh_bulge]])
                    ) / (n_disk * disk_ratio + n_bulge)
                    if p == 0:
                        offset = 0.02
                    else:
                        offset = 0.1
                    # CHECK THESE HISTOGRAMS!
                    px_d, grid = np.histogram(
                        prop[p, z, bc, mwanalogs[wh_good][wh_disk]],
                        range=(
                            min(prop[p, z, bc, mwanalogs[wh_good]]) - offset,
                            max(prop[p, z, bc, mwanalogs[wh_good]]) + offset,
                        ),
                        bins=n_bins,
                    )
                    px_b, grid = np.histogram(
                        prop[p, z, bc, mwanalogs[wh_good][wh_bulge]],
                        range=(
                            min(prop[p, z, bc, mwanalogs[wh_good]]) - offset,
                            max(prop[p, z, bc, mwanalogs[wh_good]]) + offset,
                        ),
                        bins=n_bins,
                    )
                    # Normalize to have unity area under the curve
                    px_d = (px_d.astype(float)) / scipy.integrate.simps(
                        px_d.astype(float), grid[:-1]
                    )
                    px_b = (px_b.astype(float)) / scipy.integrate.simps(
                        px_b.astype(float), grid[:-1]
                    )

                    c_x = np.zeros_like(grid[:-1])
                    for g in range(1, np.size(grid[:-1])):
                        c_x[g] = scipy.integrate.simps(
                            (
                                (px_d[0 : g + 1] * f_disk)
                                + (px_b[0 : g + 1] * (1.0 - f_disk))
                            ),
                            grid[0 : g + 1],
                        )

                    # smooth out the CDF and interpolate the appropriate values
                    function = scipy.interpolate.interp1d(
                        convolve(c_x, Box1DKernel(25)), grid[:-1]
                    )
                    lse_prop[p, z, bc] = mean_prop[p, z, bc] - function(0.16)
                    hse_prop[p, z, bc] = function(0.84) - mean_prop[p, z, bc]
        print(path_outputs)
        # save the raw statistics and the samples used
        np.savez(
            path_outputs / "raw_props.npz",
            mean_prop=mean_prop,
            lse_prop=lse_prop,
            hse_prop=hse_prop,
            wh_good=wh_good,
            wh_disk=wh_disk,
            wh_bulge=wh_bulge,
            f_disk=f_disk,
            disk_ratio=disk_ratio,
        )

    if eddbias:
        starttime = time.time()

        try:
            mean_prop
        except NameError:
            try:
                values = np.load(path_outputs / "raw_props_z0.09.npz")
            except IOError:
                print("No mean_prop file! Run again with raw_props=True")
            else:
                mean_prop = values["mean_prop"]

        ngals = int(galaxies.shape[0])
        pert_prop = np.zeros((3, 2, 10, 4, n_iters))

        # In the following loop calculate the color and luminosities after perturbing the 
        # galaxy property values by random gaussian noise in the same way as normal
        for d in range(4):
            for i in range(n_iters):
                if (i % 100) == 0:
                    print(
                        "Eddbias: %f %% Complete at: %f seconds."
                        % (
                            (100.0 * (d / 4.0) + 25.0 * (i / (n_iters - 1.0))),
                            (time.time() - starttime),
                        )
                    )
                # Perturb the sample
                galaxies_pert = galaxies.copy(deep=True)
                for column in galaxies_pert:
                    galaxies_pert[column] = galaxies_pert[column] +\
                        np.random.standard_normal((ngals,))* galaxies_sigma[column].values * (d >= 0) +\
                        np.random.standard_normal((ngals,))* galaxies_sigma[column].values * (d >= 1) +\
                        np.random.standard_normal((ngals,))* galaxies_sigma[column].values * (d >= 2) +\
                        np.random.standard_normal((ngals,))* galaxies_sigma[column].values * (d == 3) 

                # Construct the tree
                search_space = np.zeros((galaxies_pert.shape[0], galaxies_pert.shape[1]))
                for j, k in enumerate(galaxies_pert.columns):
                    search_space[:, j] = (
                        galaxies_pert[k].values - mw.at["mean", k]
                    ) / mw.at["sigma", k]
                tree = spatial.cKDTree(search_space)
                #Select analogs
                mwanalogs_pert = []
                while np.size(mwanalogs_pert) < n_mw:
                    mw_realization = mw.copy(deep=True)
                    mw_realization.loc["point"] = [
                        np.random.normal(loc=it[0], scale=it[1]) for it in (mw.T.values)
                    ]
                    mwa_index_i = mw_analog_auto(
                        tree, mw_realization, silent=True, equal_weight=False
                    )
                    mwanalogs_pert.append(mwa_index_i)
                mwanalogs_pert = np.array(mwanalogs_pert)

                wh_good, n_good, wh_disk, n_disk, wh_bulge, n_bulge, disk_ratio, f_disk = get_index_bulge_disk(
                    matched_catalog, mwanalogs_pert
                )

                for z in range(2):
                    for bc in range(10):
                        for p in range(3):
                            pert_prop[p, z, bc, d, i] = (
                                n_disk
                                * disk_ratio
                                * hlmean(
                                    prop[p, z, bc, mwanalogs_pert[wh_good][wh_disk]]
                                )
                                + n_bulge
                                * hlmean(
                                    prop[p, z, bc, mwanalogs_pert[wh_good][wh_bulge]]
                                )
                            ) / (n_disk * disk_ratio + n_bulge)
        # Now calculate the bias and associated error for each property/redshift
        xx = np.arange(2, 6, 1)
        delta_mean = np.zeros((3, 2, 10, 4))
        delta_sigma = np.zeros((3, 2, 10, 4))
        coeffs = np.zeros((3, 2, 10, 3))
        bias_sigma = np.zeros((3, 2, 10))
        bias_mean = np.zeros((3, 2, 10))
        hlmeans = np.zeros((3, 2, 10, 4, n_hlmean_boots))

        for p in range(3):
            for z in range(2):
                for b in range(10):
                    for d in range(4):
                        boots = bootstrap(pert_prop[p, z, b, d, :], n_hlmean_boots)
                        for h in range(n_hlmean_boots):
                            hlmeans[p, z, b, d, h] = hlmean(boots[:, h])

                        if d == 0:
                            delta_mean[p, z, b, d] = (
                                np.mean(hlmeans[p, z, b, d, :], axis=0)
                                - mean_prop[p, z, b]
                            )
                            delta_sigma[p, z, b, d] = np.sqrt(2.0) * np.std(
                                hlmeans[p, z, b, d, :], axis=0
                            )
                        else:
                            delta_mean[p, z, b, d] = np.mean(
                                hlmeans[p, z, b, d, :], axis=0
                            ) - np.mean(hlmeans[p, z, b, d - 1, :], axis=0)
                            delta_sigma[p, z, b, d] = np.sqrt(
                                np.std(hlmeans[p, z, b, d, :], axis=0) ** 2.0
                                + np.std(hlmeans[p, z, b, d - 1, :], axis=0) ** 2.0
                            )

                    def func(x, a, b, c):
                        return a * x ** 2 + b * x + c

                    coeffs[p, z, b, :], covar = scipy.optimize.curve_fit(
                        func, xx, delta_mean[p, z, b, :], sigma=delta_sigma[p, z, b, :]
                    )

                    # since modeling bias at the n=1 point, this just becomes the sum of the coefficients
                    bias_mean[p, z, b] = np.sum(coeffs[p, z, b, :])
                    # for same reason the total error will be the sum of the covariance matrix elements
                    bias_sigma[p, z, b] = np.sqrt(np.sum(covar))

        # ANY BIG CHANGES, MAKE SURE A QUADRATIC CURVE IS A DECENT FIT TO THE SET DELTA VALUES

        # save the eddington bias statistics separately
        np.savez(
            "eddbias.npz",
            pert_prop=pert_prop,
            hlmeans=hlmeans,
            delta_mean=delta_mean,
            delta_sigma=delta_sigma,
            coeffs=coeffs,
            bias_mean=bias_mean,
            bias_sigma=bias_sigma,
        )

    # Calculate the corrected photometric properties with the bias subtracted from the raw statistics
    if corrected_props:
        try:
            mean_prop, hse_prop, lse_prop
        except NameError:
            try:
                values = np.load(path_outputs / "raw_props_z0.09.npz")
            except IOError:
                print("No raw_props file! Run again with raw_props=True")
            else:
                mean_prop = values["mean_prop"]
                hse_prop = values["hse_prop"]
                lse_prop = values["lse_prop"]

        try:
            bias_mean
        except NameError:
            try:
                values = np.load(path_outputs / "eddbias_z0.09.npz")
            except IOError:
                print("No eddbias file! Run again with eddbias=True")
            else:
                bias_mean = values["bias_mean"]
                bias_sigma = values["bias_sigma"]

        corr_mean_prop = mean_prop - bias_mean
        corr_hse_prop = np.sqrt(hse_prop ** 2 + bias_sigma ** 2)
        corr_lse_prop = np.sqrt(lse_prop ** 2 + bias_sigma ** 2)

        np.savez(
            "corr_prop_z0.09.npz",
            corr_mean_prop=corr_mean_prop,
            corr_hse_prop=corr_hse_prop,
            corr_lse_prop=corr_lse_prop,
        )

    # Calculate the derivative of photometric properties w.r.t. SFR and M_star by offsetting the MW's
    # values by sigma/10 in both directions, giving 3 point curve.
    # We assume Eddington bias does not affect these derivatives.
    if derivatives:
        starttime = time.time()

        mw_offset = pd.DataFrame(index=["-delta", "0", "delta"], columns=mw.columns)
        mw_offset = mw_offset.fillna(0)
        for column in mw:
            mw_offset.loc["-delta", column] = -mw.at["sigma", column] / 10.0
            mw_offset.loc["delta", column] = mw.at["sigma", column] / 10.0
        # Arrays to hold properties marked by
        # [x/y/etc (property), property value, color/lum., z=0/z=0.1, index/band]
        prop_after_offset = np.zeros((mw_offset.shape[1], mw_offset.shape[0], 3, 2, 10))
        prop_derivs = pd.DataFrame(index=["derivative_array"], columns=mw.columns)
        prop_derivs = prop_derivs.fillna(0)

        # First offset the properties
        for ms, ms_col in enumerate(mw_offset.columns):
            for val, val_row in enumerate(mw_offset.index):
                print(
                    "derivs %f %% complete at %f seconds"
                    % (
                        (100.0 * (ms / 2.0) + 50.0 * (val / 3.0)),
                        (time.time() - starttime),
                    )
                )
                # Construct the tree
                search_space = np.zeros((galaxies.shape[0], galaxies.shape[1]))
                for j, k in enumerate(galaxies.columns):
                    search_space[:, j] = (
                        galaxies[k].values - mw.at["mean", k]
                    ) / mw.at["sigma", k]
                tree = spatial.cKDTree(search_space)
                mwanalogs_offset = []
                while np.size(mwanalogs_offset) < n_mw:
                    mw_realization = mw.copy(deep=True)
                    for col in mw:
                        if col == ms_col:
                            mw_realization.at["point", col] = np.random.normal(
                                loc=mw_realization.at["mean", col]
                                + mw_offset.at[val_row, ms_col],
                                scale=mw_realization.at["sigma", col],
                            )
                        else:
                            mw_realization.at["point", col] = np.random.normal(
                                loc=mw_realization.at["mean", col],
                                scale=mw_realization.at["sigma", col],
                            )
                    # mw_realization.loc["point"] = [np.random.normal(
                    #    loc=it[0]+mw_offset.at[val_row,ms_col], scale=it[1]) for it in (mw.T.values)]
                    mwa_index_i = mw_analog_auto(
                        tree, mw_realization, silent=True, equal_weight=False
                    )

                    mwanalogs_offset.append(mwa_index_i)
                mwanalogs_offset = np.array(mwanalogs_offset)
                wh_good, n_good, wh_disk, n_disk, wh_bulge, n_bulge, disk_ratio, f_disk = get_index_bulge_disk(
                    matched_catalog, mwanalogs_offset
                )

                prop_derivatives = np.zeros((3, 2, 10))
                for p in range(3):
                    for z in range(2):
                        for bc in range(10):
                            prop_after_offset[ms, val, p, z, bc] = (
                                n_disk
                                * disk_ratio
                                * hlmean(
                                    prop[p, z, bc, mwanalogs_offset[wh_good][wh_disk]]
                                )
                                + n_bulge
                                * hlmean(
                                    prop[p, z, bc, mwanalogs_offset[wh_good][wh_bulge]]
                                )
                            ) / (n_disk * disk_ratio + n_bulge)

                            if val == mw_offset.shape[1]:
                                dy = deriv(
                                    mw_offset[ms_col].values + mw.at["mean", ms_col],
                                    prop_after_offset[ms, :, p, z, bc],
                                )

                                prop_derivatives[p, z, bc] = dy[1]

            prop_derivs[ms_col] = [prop_derivatives]

        prop_derivs.to_pickle("prop_derivs_z0.09")

    if axistest:
        # Test different rejections of disks/spheriodal galaxies by calculating mean props in 3 ways:
        # 0 - Rejecting all galaxies by inclination
        # 1 - Rejecting only disk galaxies by inclination
        # 2 - Same as 1, but calc. the weighted mean with disks weighted by their fraction before cut
        starttime = time.time()

        M_r = matched_catalog["M_r"]
        gmr = matched_catalog["gmr"]
        ab_val = (np.arange(0, 19, 1) / 18.0) * 0.9  # lower bounds from b/a = 0 to 0.9
        mean_hlmean_gmr = np.zeros((3, 19))
        stddev_hlmean_gmr = np.zeros((3, 19))
        mean_hlmean_M_r = np.zeros((3, 19))
        stddev_hlmean_M_r = np.zeros((3, 19))
        n_gals = np.zeros((2, 19))

        for i in range(19):
            print(
                "axistest %s %% complete at %f seconds"
                % (str(100.0 * (i / 19.0)), (time.time() - starttime))
            )

            cut0 = np.where(matched_catalog[mwanalogs]["AB_EXP"] > ab_val[i])
            nmwcut0 = np.size(cut0)
            cut1 = np.where(
                np.logical_or(
                    (matched_catalog[mwanalogs]["AB_EXP"] > ab_val[i]),
                    (matched_catalog[mwanalogs]["FRACPSF"] > 0.5),
                )
            )
            nmwcut1 = np.size(cut1)

            n_gals[0, i] = nmwcut0
            n_gals[1, i] = nmwcut1

            # calculate the mean and error bars for each cut type via bootstrap techniques
            # g-r
            samples = bootstrap(gmr[mwanalogs][cut0], n_boots)
            hlmeans = np.zeros((n_boots))
            for b in range(n_boots):
                hlmeans[b] = hlmean(samples[:, b])
            mean_hlmean_gmr[0, i] = np.mean(hlmeans)
            stddev_hlmean_gmr[0, i] = np.std(hlmeans)

            samples = bootstrap(gmr[mwanalogs][cut1], n_boots)
            hlmeans = np.zeros((n_boots))
            for b in range(n_boots):
                hlmeans[b] = hlmean(samples[:, b])
            mean_hlmean_gmr[1, i] = np.mean(hlmeans)
            stddev_hlmean_gmr[1, i] = np.std(hlmeans)

            # calculate the weighted mean (our standard method)
            # bootstrap manually since we need subsamples of each sample
            hlmeans = np.zeros((n_boots))
            for b in range(n_boots):
                bootind = np.floor(np.random.random((nmwcut1)) * nmwcut1)
                bootind = bootind.astype("int")
                wh_disk = np.where(
                    matched_catalog[mwanalogs][cut1][bootind]["FRACPSF"] <= 0.5
                )
                n_disk = np.size(wh_disk)
                wh_bulge = np.where(
                    matched_catalog[mwanalogs][cut1][bootind]["FRACPSF"] > 0.5
                )
                n_bulge = np.size(wh_bulge)
                disk_ratio = np.true_divide(
                    np.size(np.where(matched_catalog[mwanalogs]["FRACPSF"] <= 0.5)),
                    n_disk,
                )
                # g-r
                hlmeans[b] = (
                    n_disk * disk_ratio * hlmean(gmr[mwanalogs][cut1][bootind][wh_disk])
                    + n_bulge * hlmean(gmr[mwanalogs][cut1][bootind][wh_bulge])
                ) / (n_disk * disk_ratio + n_bulge)

            mean_hlmean_gmr[2, i] = np.mean(hlmeans)
            stddev_hlmean_gmr[2, i] = np.std(hlmeans)

            # M_r
            samples = bootstrap(M_r[mwanalogs][cut0], n_boots)
            hlmeans = np.zeros((n_boots))
            for b in range(n_boots):
                hlmeans[b] = hlmean(samples[:, b])
            mean_hlmean_M_r[0, i] = np.mean(hlmeans)
            stddev_hlmean_M_r[0, i] = np.std(hlmeans)

            samples = bootstrap(M_r[mwanalogs][cut1], n_boots)
            hlmeans = np.zeros((n_boots))
            for b in range(n_boots):
                hlmeans[b] = hlmean(samples[:, b])
            mean_hlmean_M_r[1, i] = np.mean(hlmeans)
            stddev_hlmean_M_r[1, i] = np.std(hlmeans)

            # calculate the weighted mean (our standard method)
            # bootstrap manually since we need subsamples of each sample
            hlmeans = np.zeros((n_boots))
            for b in range(n_boots):
                bootind = np.floor(np.random.random((nmwcut1)) * nmwcut1)
                bootind = bootind.astype("int")
                wh_disk = np.where(
                    matched_catalog[mwanalogs][cut1][bootind]["FRACPSF"] <= 0.5
                )
                n_disk = np.size(wh_disk)
                wh_bulge = np.where(
                    matched_catalog[mwanalogs][cut1][bootind]["FRACPSF"] > 0.5
                )
                n_bulge = np.size(wh_bulge)
                disk_ratio = np.true_divide(
                    np.size(np.where(matched_catalog[mwanalogs]["FRACPSF"] <= 0.5)),
                    n_disk,
                )
                # g-r
                hlmeans[b] = (
                    n_disk * disk_ratio * hlmean(M_r[mwanalogs][cut1][bootind][wh_disk])
                    + n_bulge * hlmean(M_r[mwanalogs][cut1][bootind][wh_bulge])
                ) / (n_disk * disk_ratio + n_bulge)
            mean_hlmean_M_r[2, i] = np.mean(hlmeans)
            stddev_hlmean_M_r[2, i] = np.std(hlmeans)

        np.savez(
            "axistest_z0.09.npz",
            ab_val=ab_val,
            n_gals=n_gals,
            mean_hlmean_gmr=mean_hlmean_gmr,
            stddev_hlmean_grm=stddev_hlmean_gmr,
            mean_hlmean_M_r=mean_hlmean_M_r,
            stddev_hlmean_M_r=stddev_hlmean_M_r,
        )


# mw analogs obtained from mw_analog_auto
# path_analogs = Path.home() / "MW_morphology" / "Analogs"
# path_analogs = "/home/cef41/MW_morphology/Analogs"
# mwanalogs = readsav("/home/cef41/MW_morphology/Analogs/mwanalogs.sav")['mwanalogs']
