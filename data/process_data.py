import numpy as np
import pandas as pd

def calculate_wind_speed(u_component, v_component):
    """
    Calculate the wind speed given the u and v components of the wind.

    Parameters:
    u_component (float): The u component of the wind (east-west direction) in meters per second.
    v_component (float): The v component of the wind (north-south direction) in meters per second.

    Returns:
    float: The wind speed in miles per hour (mph).
    """
    wind_speed_m_s = np.sqrt(u_component**2 + v_component**2)
    wind_speed_mph = wind_speed_m_s * 2.23694  # Convert m/s to mph
    return wind_speed_mph

def calculate_skiable_days(snow_depth, wind_speed, min_snow_depth=0.25, max_wind_speed=30):
    """
    Calculate the number of skiable days per year based on snow depth and wind speed.

    Parameters:
    snow_depth (pd.Series): A pandas Series representing the snow depth measurements.
    wind_speed (pd.Series): A pandas Series representing the wind speed measurements.
    min_snow_depth (float, optional): The minimum snow depth required for a day to be skiable. Default is 0.25.
    max_wind_speed (float, optional): The maximum wind speed allowed for a day to be skiable. Default is 30.

    Returns:
    pd.Series: A pandas Series with the number of skiable days per year.
    """
    skiable = (snow_depth > min_snow_depth) & (wind_speed <  max_wind_speed)
    return skiable.resample('Y').sum()  # Sum skiable days per year


def calculate_mean_anomalies(df, baseline_start, baseline_end, months=None, annual=False):
    """
    Calculate mean anomalies for specified months or annually.

    Parameters:
    df (DataFrame): Pandas DataFrame with DateTime index and climate data.
    baseline_start (int): Start year for the baseline period.
    baseline_end (int): End year for the baseline period.
    months (list, optional): List of integers representing months to include in the analysis.
                             Default is None, which calculates annual anomalies if annual is True.
    annual (bool): Whether to calculate anomalies for the entire year. If False and months is None,
                   defaults to winter months (December, January, February).

    Returns:
    DataFrame: Anomalies calculated against the baseline period.
    """
    if months is not None:
        df_selected = df[df.index.month.isin(months)]
    elif annual:
        df_selected = df
    else:
        df_selected = df[df.index.month.isin([12, 1, 2])]  # Default to winter months if nothing specified

    # Group by year and calculate mean for each year
    df_mean = df_selected.groupby(df_selected.index.year).mean()

    # convert baseline start and end to datetime objects
    baseline_start = pd.to_datetime(str(baseline_start))
    baseline_end = pd.to_datetime(str(baseline_end))

    # convert the index to datetime objects
    df_mean.index = pd.to_datetime(df_mean.index, format='%Y')
    
    # Calculate baseline mean
    baseline = df_mean[(df_mean.index >= baseline_start) & (df_mean.index <= baseline_end)]
    baseline_avg = baseline.mean()
    
    # Calculate anomalies as the difference from the baseline average
    anomalies = df_mean - baseline_avg
    return anomalies
