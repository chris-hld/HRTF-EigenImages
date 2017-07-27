"""Some useful helpers for HRTF ML project."""

import numpy as np
import pandas as pd


def select_direction(azimuth, elevation, df, grid_df):
    """Get dataframe slice for one direction (azimuth and elevation).

    Directions specified in grid dataframe.

    Parameters
    ----------
    azimuth : int
        Azimuth in degrees
    elevation : int
        Elevation in degrees
    df : pandas.dataframe
        Dataframe containing measurements per person
    grid_df: pandas.dataframe
        Dataframe containing available dagrees

    Returns
    -------
    pandas.dataframe
        Slice of df for given direction

    Example
    -------
    ild_direction = select_direction(90, 0, ild, grid)

    """
    return df.loc[:, (grid_df.azimuth == azimuth) &
                  (grid_df.elevation == elevation)]
