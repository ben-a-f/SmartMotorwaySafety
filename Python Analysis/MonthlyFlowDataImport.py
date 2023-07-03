# Imports daily data from WebTRIS API. This includes both speed and flow.
# WARNING: This takes a long time to run, and will overwrite existing data.
# Remember to back up the "SpeedDataImport_7days.csv" file if you wish to keep the old version.
# The requested date will also have to be updated as the available date range is fixed at 10 years.

import pandas as pd
import numpy as np
import requests
import time

st = time.time()
# Read list of MIDAS site IDs from local file.
segmentDataPath = "SegmentIDLookup.csv"
segmentIDLookup = pd.read_csv(segmentDataPath)
segmentIDLookup = segmentIDLookup.astype({"SiteID": "str"})
siteIDList = segmentIDLookup["SiteID"].values.tolist()

# Get data from WebTRIS API.
# API is severely limited by the number of rows it can return, so requests are segmented by year.
dateSegments = ["&start_date=03032013&end_date=31122013&page=1&page_size=36000",
                "&start_date=01012014&end_date=31122014&page=1&page_size=36000",
                "&start_date=01012015&end_date=31122015&page=1&page_size=36000",
                "&start_date=01012016&end_date=31122016&page=1&page_size=36000",
                "&start_date=01012017&end_date=31122017&page=1&page_size=36000",
                "&start_date=01012018&end_date=31122018&page=1&page_size=36000",
                "&start_date=01012019&end_date=31122019&page=1&page_size=36000",
                "&start_date=01012020&end_date=31122020&page=1&page_size=36000",
                "&start_date=01012021&end_date=31122021&page=1&page_size=36000"]

APIData = {}
for i in range(0, len(siteIDList)):
    for j in range(0, len(dateSegments)):
        siteIDString = str(siteIDList[i])
        apiString1 = "https://webtris.highwaysengland.co.uk/api/v1.0/reports/Daily?sites="
        apiString2 = dateSegments[j]
        apiString = apiString1 + siteIDString + apiString2
        # Requesting blank dataset returns a decode error which inherits from ValueError
        try:
            flowData_raw = requests.get(apiString).json()
        except ValueError:
            continue
        flowData = pd.json_normalize(flowData_raw["Rows"])
        # Convert JSON to dataframe and keep relevant fields.
        flowData = flowData.loc[flowData["Total Volume"] != "", ["Report Date", "Avg mph", "Total Volume"]]
        # Formatting
        # Set column types.
        flowData.insert(loc=0, column="SiteID", value=siteIDString)
        flowData["Avg mph"] = flowData["Avg mph"].replace("", np.nan)
        flowData["Total Volume"] = flowData["Total Volume"].replace("", np.nan)
        # Using Int64 to allow for integer NaN values.
        flowData = flowData.astype({"SiteID": "str",
                                    "Avg mph": "Int64",
                                    "Total Volume": "Int64"})
        flowData["Report Date"] = flowData["Report Date"].str.replace("T00:00:00", "")
        flowData["Report Date"] = pd.to_datetime(flowData["Report Date"], format="%Y-%m-%d")
        flowData["Month"] = flowData["Report Date"].to_numpy().astype("datetime64[M]")
        # Keep only full days of data. Rows with no data may or may not be included, so remove to be sure.
        flowData = flowData.dropna(subset=["Avg mph", "Total Volume"])
        flowData["IntervalCount"] = flowData.groupby("Report Date")["Report Date"].transform("size")
        flowData = flowData.loc[flowData["IntervalCount"] == 96, :].copy()
        # Keep only months with at least 7 days of data.
        flowData["DayCount"] = flowData.groupby("Month")["Report Date"].transform('nunique')
        flowData = flowData.loc[flowData["DayCount"] >= 7, :].copy()
        # Calculate average daily flow
        flowData["DailyFlow"] = flowData.groupby("Report Date")["Total Volume"].transform("sum")
        # Calculate Average Daily Traffic and Average Speed for each month.
        flowData["FlowSpeedProduct"] = flowData["Avg mph"] * flowData["Total Volume"]
        flowData["AverageSpeed"] = flowData.groupby("Month")["FlowSpeedProduct"].transform("sum") / \
                                   flowData.groupby("Month")["Total Volume"].transform("sum")
        flowData = flowData.loc[:,
                   ['SiteID', 'Report Date', 'Month', 'DailyFlow', 'AverageSpeed']].drop_duplicates().reset_index()
        # Using ADT instead of raw monthly figures as many months are missing days/hours.
        flowData["ADT"] = flowData.groupby("Month")["DailyFlow"].transform(np.mean)
        flowData = flowData.loc[:, ["SiteID", "Month", "ADT", "AverageSpeed"]].drop_duplicates().reset_index(
            drop=True)
        flowData["SiteID"] = siteIDString
        if flowData.shape[0] != 0:
            APIData[i*9 + j] = flowData

flowData = pd.concat(APIData.values(), ignore_index=True)

# Join smart motorway segment info to flow data.
flowData = flowData.merge(segmentIDLookup.loc[:, ["SiteID", "SegmentDir", "Type", "StartDate", "SegLength"]],
                          how="left",
                          on=["SiteID"])

elapsed_time = time.time() - st

flowData.to_csv("SpeedDataImport_7days.csv", index=False)