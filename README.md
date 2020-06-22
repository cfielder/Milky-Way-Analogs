# Selecting SDSS MW Analogs
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
![GitHub last commit](https://img.shields.io/github/last-commit/cfielder/MW_Morphology.svg)

Code that allows for selection of a Milky Way anlogs sample in various parameter spaces and analysis
of said sample.

## Installing

### Directly from Repository

`git clone https://github.com/cfielder/MW_Morphology`

## Usage

This code has been sepcifically built to work around a cleaned sample. For example in Licquia et al. 2015
the sample contains all objects in SDSS-III DR8 and MPA-JHU of which a significant portion was then discarded 
due to various flags. Of these a volume-limited sample is then selected. This is the sample going into these 
scripts for analysis.

If you want to work with this specific sample please contact us.

## Description of Scripts

To make a sample (default 5,000) of analogs, values are randomly drawn from the fiducial MW PDF in the parameter space
of interest (minimum of two parameters). For each set of values, the nearest neighbors are found in a binary tree
constructed from the volume-limited sample. 

Then the sample's photometric properties, derivatives of the photometric properties, and the errors + Eddington bias are 
calculated.

For ease of use of all of these scripts, we suggest using a script that has the dataframes of your galaxies and Milky Way
parameters. This can also then easily run any of the scripts that you are interested in. Refer to `mstar_sfr.py` as an 
example. 

### Step 1:
- **Make a cross matched catalog and prop array**
  - vollim_catalog.py and vollim_simard.py are two examples with which to make cross matches. vollim_catalog.py must be 
    run before vollim_simard.py. These scripts access the catalogs being used, so please update Line 15 in load_catalogs.py
    to point to the directory where your catalogs are saved. Line 12 in vollim_simard.py must also be updated to load in
    your volume-limited SDSS dataframe created from vollim_catalog.py.
  - These scripts make the base dataframe that Milky Way analogs are selected from/analysed and makes the prop array necessary 
    for running mw_color_lum_stats by utilizing the `make_prop_array()` function (this script does not need to be modified). 
    `make_prop_array` takes in the cross matched kcorrect and mass_to_light fits tables, as well as the number of galaxies in
    said cross-matched subsample.
  - Further cross-matches can be made, but we suggest doing this in layers instead of trying to do multiple cross-matches
    at once. 
  - It is useful to have a separate subdirectory from your working directory that contains said catalogs/arrays.

### Step 2:
- **Run `select_analogs.py`** 
  - The `select_analogs()` function takes in a dataframe of the cross-matched sample of galaxies and a dataframe of the 
  Milky Way morphological parameters. The columns of each dataframe correspond to a parameter of interest. This function 
  accesses `mw_analog_auto.py` which contains a function for selecting a single analog from the binary tree.
- **Save the Output**
  - Save this output as a numpy array for convenince. This array of analog indices is necessary for further analysis.
  
  Example:
  ```
  import pandas as pd 
  import numpy as np
  from pathlib import Path
  from select_analogs.py import select_analogs
  
  catalogs = Path.home() / "MW_morphology" / "Catalogs"
  cross_matched_catalog = catalogs / "Simard_Cross_match")
  galaxies = cross_matched_catalog[["logmass","sfr"]].copy(deep=True)
  galaxies.rename(columns={"logmass":"x"},inplace=True)
  galaxies.rename(columns={"sfr":"y"},inplace=True)
  galaxies_sigma = cross_matched_catalog[["sigma_logmass","sigma_sfr"]].copy(deep=True)
  galaxies_sigma.rename(columns={"sigma_logmass":"x"},inplace=True)
  galaxies_sigma.rename(columns={"sigma_sfr":"y"},inplace=True)
  sfr = [mw_mean_sfr, mw_sigma_sfr]
  mw = pd.DataFrame({"x": mstar, "y": sfr}, index=["mean","sigma"])
  
  mwanalogs = select_analogs(galaxies,mw)
  np.save("mstar_sfr_analogs",mwanalogs)
  ```
- **NOTE**
  - The column labels of the dataframes do not matter, as long as they match (e.g. `"x"` and `"y"` in this case). The index
    labels of the `mw` dataframe MUST be `"mean"` and `"sigma"`.
    
### Step 3 (Optional):
- You can plot the PDF of the Milky Way and the analog sample you selected using the `mw_analog_histograms.py` script. 
  If this is your first time drawing a Milky Way analog sample we suggest checking these histograms to ensure that the sample
  you have selected is correct and matches the PDF of the Milky Way.
  
### Step 4:
- **Update paths**
  - Update the path where you want the output to be saved in `mw_color_lum_stats.py` on line  218 if necessary. Defaults to 
    your `home/user/MW_morphology`
- **Run `mw_color_lum_stats.py`**
  - This code does the analysis of your analogs! 
    It looks up the magnitude/color/mass_to_light ratio, redshift (0 or 0.1), and bandpass/color index of the volume limited      
    sample.
  - `raw_props` (Takes several minutes to run)
    Mean and error of photometric properties are calculated. Edge-on and disk-dominated galaxies are cut before this 
    calculation (same of eddington bias and derivatives calculations).
  - `eddbias` (Takes ~2 hours to run)
    Perturb galaxy properties 100 times at 4 noise levels in order to calculated the Eddington bias.
    Requires that raw_props has run. 
  - `corrected_props` (Takes ~ 1 min to run)
    Eddington bias is subtracted off the mean and error of the photometric properties.
    Requires that raw_props and eddbias have run.
  - `derivs` (Takes about 80 seconds to run)
    The derivatives of the various photometric properties are calculated. 
  - `axistest` (Takes ~3 hours to run)
    The impact of the effect of cutting edge-on disk-dominated objects is tested by varying the minimum allowed axis 
    ratios for disk-dominated galaxies. 
    
- **Using the Script**
  - The function `mw_color_lum_stats()` requires the same galaxy sample dataframe as Step 2, an additional 
    dataframe containing the errors of that galaxy sample, the same dataframe of the Milky Way morphological 
    parameters, the array of indices that contains the Milky Way analogs sample selected in Step 2, the cross 
    matched catalog of the parameters of interest generated in Step 1, and the properties array generated in 
    Step 1.
    
    Example:
    ```
    import pandas as pd 
    import numpy as np
    from pathlib import Path
    from mw_color_lum_stats.py import *
    
    catalogs = Path.home() / "MW_morphology" / "Catalogs"
    cross_matched_catalog = catalogs / "Simard_Cross_match")
    prop = np.load(catalogs / "prop_simard.npy")
    mwanalogs = np.load("/home/username/mwanalogs.npy")
    galaxies = cross_matched_catalog[["logmass","sfr"]].copy(deep=True)
    galaxies.rename(columns={"logmass":"x"},inplace=True)
    galaxies.rename(columns={"sfr":"y"},inplace=True)
    galaxies_sigma = cross_matched_catalog[["sigma_logmass","sigma_sfr"]].copy(deep=True)
    galaxies_sigma.rename(columns={"sigma_logmass":"x"},inplace=True)
    galaxies_sigma.rename(columns={"sigma_sfr":"y"},inplace=True)
    mstar = [mw_mean_log_mstar, mw_sigma_log_mstar]
    sfr = [mw_mean_sfr, mw_sigma_sfr]
    mw = pd.DataFrame({"x": mstar, "y": sfr}, index=["mean","sigma"])
  
    mw_color_lum_stats(
        galaxies,
        galaxies_sigma,
        mw,
        mwanalogs,
        matched_catalog,
        prop,
        raw_props=True,
        eddbias=True,
        corrected_props=True,
        derivative=True,
        axistest=True
        )
    ```
 - **Note**
   - Each optional module saves its output.
   
### Optional Functional Method For Steps 2-4:
In `do_full_analysis.py`, there is a function called `do_full_analysus()`. This 
function contains options for running steps 2-4 for you. Use as follows:
- **Update paths**
  - Update the `catalogs` path on Line 59 to point to where you have saved your analogs if they have been calculated.
  - Update the patsh described in the previous steps as well, as these scripts are still accessed.
- **Run `do_full_analysis.py`**
  - Depending what options you set to `True`, this function will select analogs, plot them, and do 
    full photometry analysis. All outputs will be saved. 
  - We recommend having a script that reads in your cross matched data and Milky Way parameters and then runs this
    function.
-**Example Wrapper**
  - `mstar_sfr.py` is an example wrapper script. This scrip selects analogs in mass and star formation rate, plots them,
     and runs the photometric analysis. 
  - Step 1 MUST be completed before this script can be executed.

## Authors

* **Catherine Fielder** - *Python Translation and Overhaul* 

* **Tim Licquia** - *Initial Coding* See http://adsabs.harvard.edu/abs/2015ApJ...809...96L for details.

With additional assistance from Brett Andrews and Jeff Newman.
