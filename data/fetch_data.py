import ee
import pandas as pd

# Initialize the Earth Engine module
ee.Initialize()

class DataFetcher:
    """
    A class to fetch daily variable data from the ECMWF ERA5-LAND dataset for a specific geographic point and date range.
    Attributes:
        lat (float): Latitude of the geographic point.
        lon (float): Longitude of the geographic point.
        start_date (str): Start date for the data fetching period in 'YYYY-MM-DD' format.
        end_date (str): End date for the data fetching period in 'YYYY-MM-DD' format.
        point (ee.Geometry.Point): Earth Engine geometry point object for the specified latitude and longitude.
        start_date (str): Start date for the data fetching period.
        end_date (str): End date for the data fetching period.
    Methods:
        fetch_daily_variable(variable):
            Fetches daily data for the specified variable from the ECMWF ERA5-LAND dataset.
            Args:
                variable (str): The variable to fetch from the dataset (e.g., 'temperature_2m').
            Returns:
                pd.DataFrame: A pandas DataFrame with DateTime as the index and the specified variable as the column.
    """
    def __init__(self, lat, lon, start_date, end_date):
        self.point = ee.Geometry.Point(lon, lat)
        self.start_date = start_date
        self.end_date = end_date

    def fetch_daily_variable(self, variable):
        dataset = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
            .filterDate(self.start_date, self.end_date) \
            .select(variable) \
            .filterBounds(self.point)

        def to_feature(image):
            reduction = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.point,
                scale=11000  # Adjust scale to match dataset resolution
            )
            value = reduction.get(variable)
            return ee.Feature(None, {'value': value, 'time': image.date().format()})

        data_feat = dataset.map(to_feature)
        data_list = data_feat.reduceColumns(ee.Reducer.toList(2), ['time', 'value']).getInfo()['list']
        df = pd.DataFrame(data_list, columns=["DateTime", variable])
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        return df
    