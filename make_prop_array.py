import numpy as np


def make_prop_array(df):
    """Since we are only interested in the kcorrected colors and absmags of milky way analogs,
    concisely organize array marked by
    [absmag/color/mass_to_light, z=0/z=0.1, bandpass/color_index, object]
    This array is necessary input for mw_color_lum_stats.py to run. This is the array that 
    the statistics, etc. are calculated from.

    Args:
        df (pandas dataframe): Dataframe that has the info we are drawing from.

    Returns:
        A 2D array of the properties with which mw_color_lum_stats() will run analysis on.
        Check the documentation to be sure that the correct prop array is being loaded in 
        appropriately. We recommend that after this function is called the array is saved
        for ease of use.
    """
    prop = np.zeros((3, 2, 10, len(df)))
    # z=0 abs mags (CMODEL)
    prop[0, 0, 0, :] = df.cmodel_M_u.values
    prop[0, 0, 1, :] = df.cmodel_M_g.values
    prop[0, 0, 2, :] = df.cmodel_M_r.values
    prop[0, 0, 3, :] = df.cmodel_M_i.values
    prop[0, 0, 4, :] = df.cmodel_M_z.values
    # z=0.1 abs mags (CMODEL)
    prop[0, 1, 0, :] = df.cmodel_M_u_z0P1.values
    prop[0, 1, 1, :] = df.cmodel_M_g_z0P1.values
    prop[0, 1, 2, :] = df.cmodel_M_r_z0P1.values
    prop[0, 1, 3, :] = df.cmodel_M_i_z0P1.values
    prop[0, 1, 4, :] = df.cmodel_M_z_z0P1.values
    # z=0 abs mags (CMODEL) johnson
    prop[0, 0, 5, :] = df.cmodel_M_U.values
    prop[0, 0, 6, :] = df.cmodel_M_B.values
    prop[0, 0, 7, :] = df.cmodel_M_V.values
    prop[0, 0, 8, :] = df.cmodel_M_R.values
    prop[0, 0, 9, :] = df.cmodel_M_I.values
    # z=0.1 abs mags (CMODEL) Johnson
    prop[0, 1, 5, :] = df.cmodel_M_U_z0P1.values
    prop[0, 1, 6, :] = df.cmodel_M_B_z0P1.values
    prop[0, 1, 7, :] = df.cmodel_M_V_z0P1.values
    prop[0, 1, 8, :] = df.cmodel_M_R_z0P1.values
    prop[0, 1, 9, :] = df.cmodel_M_I_z0P1.values
    # z=0  mass to light ugriz
    prop[2, 0, 0, :] = df.mass_to_light_u.values
    prop[2, 0, 1, :] = df.mass_to_light_g.values
    prop[2, 0, 2, :] = df.mass_to_light_r.values
    prop[2, 0, 3, :] = df.mass_to_light_i.values
    prop[2, 0, 4, :] = df.mass_to_light_z.values
    # z=0.1 mass to light ugriz
    prop[2, 1, 0, :] = df.mass_to_light_u_z0P1.values
    prop[2, 1, 1, :] = df.mass_to_light_g_z0P1.values
    prop[2, 1, 2, :] = df.mass_to_light_r_z0P1.values
    prop[2, 1, 3, :] = df.mass_to_light_i_z0P1.values
    prop[2, 1, 4, :] = df.mass_to_light_z_z0P1.values
    # z=0 mass to light Johnsons
    prop[2, 0, 5, :] = df.mass_to_light_U.values
    prop[2, 0, 6, :] = df.mass_to_light_B.values
    prop[2, 0, 7, :] = df.mass_to_light_V.values
    prop[2, 0, 8, :] = df.mass_to_light_R.values
    prop[2, 0, 9, :] = df.mass_to_light_I.values
    #z=0.1 mass to light Johnsons
    prop[2, 1, 5, :] = df.mass_to_light_U_z0P1.values
    prop[2, 1, 6, :] = df.mass_to_light_B_z0P1.values
    prop[2, 1, 7, :] = df.mass_to_light_V_z0P1.values
    prop[2, 1, 8, :] = df.mass_to_light_R_z0P1.values
    prop[2, 1, 9, :] = df.mass_to_light_I_z0P1.values
    # z=0 colors ugriz (MODEL)
    prop[1, 0, 0, :] = df.model_M_u.values - df.model_M_g.values
    prop[1, 0, 1, :] = df.model_M_u.values - df.model_M_r.values
    prop[1, 0, 2, :] = df.model_M_g.values - df.model_M_r.values
    prop[1, 0, 3, :] = df.model_M_r.values - df.model_M_i.values
    prop[1, 0, 4, :] = df.model_M_i.values - df.model_M_z.values
    # z=0.1 colors ugriz(MODEL)
    prop[1, 1, 0, :] = df.model_M_u_z0P1.values - df.model_M_g_z0P1.values
    prop[1, 1, 1, :] = df.model_M_u_z0P1.values - df.model_M_r_z0P1.values
    prop[1, 1, 2, :] = df.model_M_g_z0P1.values - df.model_M_r_z0P1.values
    prop[1, 1, 3, :] = df.model_M_r_z0P1.values - df.model_M_i_z0P1.values
    prop[1, 1, 4, :] = df.model_M_i_z0P1.values - df.model_M_z_z0P1.values
    # z=0 colors Johnsons
    prop[1, 0, 5, :] = df.model_M_U.values - df.model_M_B.values
    prop[1, 0, 6, :] = df.model_M_U.values - df.model_M_V.values
    prop[1, 0, 7, :] = df.model_M_B.values - df.model_M_V.values
    prop[1, 0, 8, :] = df.model_M_V.values - df.model_M_R.values
    prop[1, 0, 9, :] = df.model_M_R.values - df.model_M_I.values
    # z=0.1 colors Johnsons
    prop[1, 1, 5, :] = df.model_M_U_z0P1.values - df.model_M_B_z0P1.values
    prop[1, 1, 6, :] = df.model_M_U_z0P1.values - df.model_M_V_z0P1.values
    prop[1, 1, 7, :] = df.model_M_B_z0P1.values - df.model_M_V_z0P1.values
    prop[1, 1, 8, :] = df.model_M_V_z0P1.values - df.model_M_R_z0P1.values
    prop[1, 1, 9, :] = df.model_M_R_z0P1.values - df.model_M_I_z0P1.values

    return prop
