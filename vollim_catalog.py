import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table, Column

from load_catalogs import *
#from make_prop_array import make_prop_array

"""This script makes cross-matched catalogues useful for data analysis with 
    Milky Way analogs.
    
The outputs of this script are necessary for running select_analogs and mw_color_lum_stats.
This script first reads in all of the catalogs of interest for the data analysis.
Be sure to update the path to you catalog directory.

Then the match_to_catalog_sky() function is used to match to various catalogs. In
this script we first match to the Simard catalogue and save outputs. Then we match
to the Samir catalogue and save the outputs. You can do this matching any number of times.

Returns:
    dataframe of the matched catalog
    2d numpy array of properties to be used in mw_color_lum_stats 
"""
######################################
# Make a catalogue from DR8 ##########
######################################
kcorrect = kcorrect[vollim]
mass_to_light = mass_to_light[vollim]
sdss = {
    "dr8_RA": dr8[vollim]["RA"],
    "dr8_Dec": dr8[vollim]["DEC"],
    "dr8_fiberid": dr8[vollim]["FIBERID"],
    "dr8_mjd": dr8[vollim]["MJD"],
    "dr8_plate": dr8[vollim]["PLATE"],
    "redshift": dr8[vollim]["Z"],
    "redshift_err": dr8[vollim]["Z_ERR"],
    "AB_EXP": dr8[vollim]["AB_EXP"][:, 2],
    "FRACPSF": dr8[vollim]["FRACPSF"][:, 2],

    "cmodel_M_u": kcorrect["CMODEL_UGRIZ_ABSMAGS_K0"][:, 0],
    "model_M_u": kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 0],
    "cmodel_M_g": kcorrect["CMODEL_UGRIZ_ABSMAGS_K0"][:, 1],
    "model_M_g": kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 1],
    "cmodel_M_r": kcorrect["CMODEL_UGRIZ_ABSMAGS_K0"][:, 2],
    "model_M_r": kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 2],
    "cmodel_M_i": kcorrect["CMODEL_UGRIZ_ABSMAGS_K0"][:, 3],
    "model_M_i": kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 3],
    "cmodel_M_z": kcorrect["CMODEL_UGRIZ_ABSMAGS_K0"][:, 4],
    "model_M_z": kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 4],
    "cmodel_M_U": kcorrect["CMODEL_UBVRI_ABSMAGS_K0"][:, 0],
    "model_M_U": kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 0],
    "cmodel_M_B": kcorrect["CMODEL_UBVRI_ABSMAGS_K0"][:, 1],
    "model_M_B": kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 1],
    "cmodel_M_V": kcorrect["CMODEL_UBVRI_ABSMAGS_K0"][:, 2],
    "model_M_V": kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 2],
    "cmodel_M_R": kcorrect["CMODEL_UBVRI_ABSMAGS_K0"][:, 3],
    "model_M_R": kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 3],
    "cmodel_M_I": kcorrect["CMODEL_UBVRI_ABSMAGS_K0"][:, 4],
    "model_M_I": kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 4],
    "gmr": kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 1]
    - kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 2],
    "umg": kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 0]
    - kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 1],
    "umr": kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 0]
    - kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 2],
    "imz": kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 3]
    - kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 4],
    "logmass": logmass[vollim]["AVG"],
    "sigma_logmass": (
        (
            logmass[vollim]["P84"]
            - logmass[vollim]["P16"]
        )
        / 2
    ),
    "sfr": logsfr[vollim]["AVG"],
    "sigma_sfr": (
        (
            logsfr[vollim]["P84"]
            - logsfr[vollim]["P16"]
        )
        / 2
    ),
}
sdss_catalog = Table(sdss)
sdss_catalog_df = pd.DataFrame(sdss)
sdss_catalog_df.to_pickle("SDSS_matched_catalog")
#prop_sdss = make_prop_array(kcorrect,mass_to_light,np.size(vollim))
#np.save("prop_sdss", prop_sdss)

#import matplotlib.pyplot as plt
#plt.hist(dr8["AB_EXP"][:,2],bins=100,range=(0,1),alpha=0.5,density=True,label="SDSS b/a")
#plt.hist(dr8[vollim]["AB_EXP"][:,2],bins=100,range=(0,1),alpha=0.5,density=True,label="Vollim b/a")
#plt.xlabel("b/a")
#plt.legend()
#plt.show()
 
