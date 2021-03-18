import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table, Column
from pathlib import Path

from load_catalogs import *
#from make_prop_array import make_prop_array

sdss = pd.read_pickle(Path.home() / "Catalogs" / "SDSS_matched_catalog")
sdss = Table.from_pandas(sdss)
######################################
# Match SDSS to Simard catalogue #####
######################################
dr8_matched_coords = SkyCoord(
    ra=sdss["dr8_RA"] * u.degree,
    dec=sdss["dr8_Dec"] * u.degree,
)
simard_coords = SkyCoord(ra=simard["_RA"] * u.degree, dec=simard["_DE"] * u.degree)
sim_idx, d2d, d3d = dr8_matched_coords.match_to_catalog_sky(simard_coords)

# restrict to RA/Dec matches between DR7 and DR8 values less than 1.5 arcsec in separatio
# restricts to matches that are less than 1.5 arcsec separated on the sky
sep_idx_simard = np.where(d2d.value < 1.5 / 3600.0)
######################################
# Make a cross-matched catalogue from
# DR8 and Simard #####################
######################################
cross_matched_catalog = sdss[sep_idx_simard]
cross_matched_catalog_df = cross_matched_catalog.to_pandas()
cross_matched_catalog_df["B_T_r"] = simard[sim_idx][sep_idx_simard]["__B_T_r"]
cross_matched_catalog_df["e_B_T_r"] = simard[sim_idx][sep_idx_simard]["e__B_T_r"]
cross_matched_catalog_df["Rd"] = simard[sim_idx][sep_idx_simard]["Rd"]
cross_matched_catalog_df["e_Rd"] = simard[sim_idx][sep_idx_simard]["e_Rd"]
cross_matched_catalog_df.to_pickle("Vollim_Simard_Cross_match")

#kcorrect = kcorrect[vollim][sep_idx_simard]
#mass_to_light = mass_to_light[vollim][sep_idx_simard]
#prop = make_prop_array(kcorrect,mass_to_light,np.size(vollim[sep_idx_simard]))
#np.save("prop_vollim_simard", prop)
#np.save("sep_idx_simard", sep_idx_simard)

print(len(cross_matched_catalog_df["dr8_RA"]))
