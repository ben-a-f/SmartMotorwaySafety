# Filters the full 2010-2021 STATS19 (accident) dataset to 03/13 - 12/21 to match available flow data.
# This dataset is then to be imported into GIS to match accidents to motorway segments.

import pandas as pd

# read STATS19 data from local file.
accidentDataPath = "..\\Accident Data\\dft-road-casualty-statistics-accident-2010-2021.csv"
accidentData = pd.read_csv(accidentDataPath, encoding="latin-1")
accidentData["date"] = pd.to_datetime(accidentData["date"], format="%d/%m/%Y")

filteredData = accidentData.loc[accidentData["date"] >= "2013-03-01", ["ï..accident_index", "location_easting_osgr",
                                                                        "location_northing_osgr", "accident_severity",
                                                                        "number_of_vehicles", "number_of_casualties",
                                                                        "date", "trunk_road_flag", "first_road_class",
                                                                       "first_road_number"]]

# Want to visually inspect the data in GIS before filtering to only confirmed m'way accidents to assess missing data/errors.
# filteredData = filteredData.loc[(filteredData["trunk_road_flag"] == 1) & (filteredData["first_road_class"] == 1)]
# POST NOTE: trunk and road class flags proved accurate, these same filters have been applied in GIS but it would be safe to apply them here.

filteredData.to_csv("..\\Accident Data\\accidents-March-2013-Dec-2021.csv")