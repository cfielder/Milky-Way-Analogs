import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from MW_params import *
from select_analogs import select_analogs
from mw_analog_histograms import plot_analogs
from mw_color_lum_stats import mw_color_lum_stats


def do_full_analysis(
    galaxies,
    galaxies_sigma,
    mw,
    matched_catalog,
    prop,
    filename,
    get_analogs=False,
    get_plots=False,
    get_photo_analysis=False,
    **kwargs
):
    """Used for selecting analogs, plotting them, and doing photometric analysis.

    If all options are set to False the function will not perform anything or 
    return anything. get_phot_analysis and get_plots requires that an analog sample
    was selected either in this function or loaded in on Line 31. You may change 
    where analogs from get_analogs are saved on Line 

    Args:
        galaxies (dataframe): A dataframe consisting of catalogs being used to 
            find analogs in. Make sure these values are matched in RA and Dec.
        galaxies_sigma (dataframe): Same as galaxies but the errors of the measurements.
        mw (dataframe): A dataframe that contains the measured Milky Way values.
        matched_catalog (pandas dataframe): The main catalog (included cross-matching 
            restriction and volume limited sample cuts already) that the analogs are 
            found in. This is generated in cross_matched_catalogs.py 
        prop (2d array): A 2D array containing the mag, color, and mass-to-light ratio of the 
            cross-matched galaxy sample.
        filename (string): The filename that you want to save your selected analogs to/load
            analogs from.
        get_analogs (bool): If set to True this generates a set of Milky Way analogs using 
            the select_analogs() function and saves the array of indices of the analogs
            in the cross matched catalog.
            If set to False this will load in a catalog on analogs previously saved from 
            select_analogs(). Please update the path if needed.
        get_plots (bool): If set to True this will plot a histogram of you analogs along with
            the PDF of the Milky Way.
        get_photo_analysis (bool): If set to True this will run the photometric analysis on 
            the Milky Way analogs.

    Returns:
        A Milky Way analog sample, plots of said sample, and the photometric analysis output,
        see mw_color_lum_stats.py for details.
    """
    if get_analogs:
        mwanalogs = select_analogs(galaxies, mw, **kwargs)
        np.save(Path.home() / filename, mwanalogs)
    if not get_analogs:
        mwanalogs = np.load(Path.home() / "MW_Morphology" / filename)
    if get_plots:
        plot_analogs(galaxies, mw, mwanalogs, save_fig=True, **kwargs)

    if get_photo_analysis:
        mw_color_lum_stats(
            galaxies,
            galaxies_sigma,
            mw,
            mwanalogs,
            matched_catalog,
            prop,
            **kwargs
        )
