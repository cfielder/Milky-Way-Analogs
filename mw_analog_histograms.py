import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_analogs(galaxies, mw, mwanalogs, save_fig=False, **kwargs):
    """Plots selected analogs along with the parameter's respective
    Milky Way probability distribution function. Check these before
    running mw_color_lum_stats() to ensure that your analogs are correct.

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
        mw_analogs (array): Array of indices of Milky Way analogs in the 
             galaxy parameter catalogs.

        save_fig (boolean): Default False. If set to True will save the 
             figures for you.

    Returns:
        An MPL figure for each parameter being studying showing a histogram
            of the analogs along with the PDF of the Milky Way. 

    """

    axis_font = {"size": "20"}
    for column in mw.columns:
        fig, ax = plt.subplots()
        low_lim = np.min(galaxies[column].values[mwanalogs])
        upper_lim = np.max(galaxies[column].values[mwanalogs])
        plt.hist(
            galaxies[column].values[mwanalogs],
            density=True,
            bins=20,
            histtype="stepfilled",
            alpha=0.5,
        )
        x = np.linspace(low_lim, upper_lim, len(mwanalogs))
        plt.plot(x, norm.pdf(x, mw.at["mean", column], mw.at["sigma", column]), "r-")
        plt.xlabel(column, **axis_font)
        plt.ylabel(r"N", **axis_font)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        if save_fig:
            plt.savefig(fname=column+".pdf", format="pdf")
        else:
            plt.show()
