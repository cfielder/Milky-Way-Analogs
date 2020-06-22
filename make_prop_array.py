import numpy as np


def make_prop_array(kcorrect, mass_to_light, size):
    """Since we are only interested in the kcorrected colors and absmags of milky way analogs,
    concisely organize array marked by
    [absmag/color/mass_to_light, z=0/z=0.1, bandpass/color_index, object]
    This array is necessary input for mw_color_lum_stats.py to run. This is the array that 
    the statistics, etc. are calculated from.

    Args:
        kcorrect (fits table): The k-corrected catalog that is generated from the SDSS DR8
            volume limited sample.
        mass_to_light (fits table): The catalog of the mass to light ratios of the SDSS DR8
            volume limited sample.
        size (int): The size of the sub-sample being used for analyzing milky way analogs. 
            It is assumed that kcorrect and mass_to_light have already been cross-matched and 
            indexed to match the the sub-sample of galaxies of interest (if any, otherwise vollim
            should be the default).

    Returns:
        A 2D array of the properties with which mw_color_lum_stats() will run analysis on.
        Check the documentation to be sure that the correct prop array is being loaded in 
        appropriately. We recommend that after this function is called the array is saved
        for ease of use.
    """
    prop = np.zeros((3, 2, 10, size))
    # z=0 abs mags (CMODEL)
    prop[0, 0, :5, :] = kcorrect["CMODEL_UGRIZ_ABSMAGS_K0"].transpose()
    # z=0.1 abs mags (CMODEL)
    prop[0, 1, :5, :] = kcorrect["CMODEL_UGRIZ_ABSMAGS_K0P1"].transpose()
    # z=0 abs mags (CMODEL) johnson
    prop[0, 0, 5:10, :] = kcorrect["CMODEL_UBVRI_ABSMAGS_K0"].transpose()
    # z=0.1 abs mags (CMODEL) Johnson
    prop[0, 1, 5:10, :] = kcorrect["CMODEL_UBVRI_ABSMAGS_K0P1"].transpose()
    # z=0  mass to light ugriz
    prop[2, 0, :5, :] = mass_to_light["UGRIZ_K0"].transpose()
    # z=0.1 mass to light ugriz
    prop[2, 1, :5, :] = mass_to_light["UGRIZ_K0P1"].transpose()
    # z=0 mass to light Johnsons
    prop[2, 0, 5:10, :] = mass_to_light["UBVRI_K0"].transpose()
    # z=0.1 mass to light Johnsons
    prop[2, 1, 5:10, :] = mass_to_light["UBVRI_K0P1"].transpose()
    # z=0 colors ugriz (MODEL)
    prop[1, 0, 0, :] = np.transpose(
        kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 0]
        - kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 1]
    )
    prop[1, 0, 1, :] = np.transpose(
        kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 0]
        - kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 2]
    )
    prop[1, 0, 2, :] = np.transpose(
        kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 1]
        - kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 2]
    )
    prop[1, 0, 3, :] = np.transpose(
        kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 2]
        - kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 3]
    )
    prop[1, 0, 4, :] = np.transpose(
        kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 3]
        - kcorrect["MODEL_UGRIZ_ABSMAGS_K0"][:, 4]
    )
    # z=0.1 colors ugriz(MODEL)
    prop[1, 1, 0, :] = np.transpose(
        kcorrect["MODEL_UGRIZ_ABSMAGS_K0P1"][:, 0]
        - kcorrect["MODEL_UGRIZ_ABSMAGS_K0P1"][:, 1]
    )
    prop[1, 1, 1, :] = np.transpose(
        kcorrect["MODEL_UGRIZ_ABSMAGS_K0P1"][:, 0]
        - kcorrect["MODEL_UGRIZ_ABSMAGS_K0P1"][:, 2]
    )
    prop[1, 1, 2, :] = np.transpose(
        kcorrect["MODEL_UGRIZ_ABSMAGS_K0P1"][:, 1]
        - kcorrect["MODEL_UGRIZ_ABSMAGS_K0P1"][:, 2]
    )
    prop[1, 1, 3, :] = np.transpose(
        kcorrect["MODEL_UGRIZ_ABSMAGS_K0P1"][:, 2]
        - kcorrect["MODEL_UGRIZ_ABSMAGS_K0P1"][:, 3]
    )
    prop[1, 1, 4, :] = np.transpose(
        kcorrect["MODEL_UGRIZ_ABSMAGS_K0P1"][:, 3]
        - kcorrect["MODEL_UGRIZ_ABSMAGS_K0P1"][:, 4]
    )
    # z=0 colors Johnsons
    prop[1, 0, 5, :] = np.transpose(
        kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 0]
        - kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 1]
    )
    prop[1, 0, 6, :] = np.transpose(
        kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 0]
        - kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 2]
    )
    prop[1, 0, 7, :] = np.transpose(
        kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 1]
        - kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 2]
    )
    prop[1, 0, 8, :] = np.transpose(
        kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 2]
        - kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 3]
    )
    prop[1, 0, 9, :] = np.transpose(
        kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 3]
        - kcorrect["MODEL_UBVRI_ABSMAGS_K0"][:, 4]
    )
    # z=0.1 colors Johnsons
    prop[1, 1, 5, :] = np.transpose(
        kcorrect["MODEL_UBVRI_ABSMAGS_K0P1"][:, 0]
        - kcorrect["MODEL_UBVRI_ABSMAGS_K0P1"][:, 1]
    )
    prop[1, 1, 6, :] = np.transpose(
        kcorrect["MODEL_UBVRI_ABSMAGS_K0P1"][:, 0]
        - kcorrect["MODEL_UBVRI_ABSMAGS_K0P1"][:, 2]
    )
    prop[1, 1, 7, :] = np.transpose(
        kcorrect["MODEL_UBVRI_ABSMAGS_K0P1"][:, 1]
        - kcorrect["MODEL_UBVRI_ABSMAGS_K0P1"][:, 2]
    )
    prop[1, 1, 8, :] = np.transpose(
        kcorrect["MODEL_UBVRI_ABSMAGS_K0P1"][:, 2]
        - kcorrect["MODEL_UBVRI_ABSMAGS_K0P1"][:, 3]
    )
    prop[1, 1, 9, :] = np.transpose(
        kcorrect["MODEL_UBVRI_ABSMAGS_K0P1"][:, 3]
        - kcorrect["MODEL_UBVRI_ABSMAGS_K0P1"][:, 4]
    )

    return prop
