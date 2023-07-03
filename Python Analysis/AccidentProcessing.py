# Processes the accident data once it has been spatially mapped to the relevant motorway segments through GIS.
# Produces monthly accident totals for each segment.

import pandas as pd

accidentSegmentPath = "AccidentSegmentLookup.csv"
accidentSegment = pd.read_csv(accidentSegmentPath)
# Discard irrelevant columns, rename truncated names (due to importing into GIS) and format date.
accidentSegment = accidentSegment.loc[:, ["ï..accide", "date", "accident_s", "number_of_", "number_o_1", 
                                          "SM", "Section", "SegmentDir"]]
accidentSegment = accidentSegment.rename(columns = {"ï..accide": "AccidentIndex",
                                                    "date": "Date",
                                                    "accident_s": "AccidentSeverity",
                                                    "number_of_": "NumberVehicles",
                                                    "number_o_1": "NumberCasualties"})
accidentSegment["Date"] = pd.to_datetime(accidentSegment["Date"], format="%Y-%m-%d")
accidentSegment["Month"] = accidentSegment["Date"].to_numpy().astype("datetime64[M]")

# Get accident totals by segment, month and severity for all segments.
monthlyAccidentSegment = accidentSegment.groupby(["SegmentDir", "Month", "AccidentSeverity"]).size().reset_index(name="Count")
monthlyAccidentSegment = monthlyAccidentSegment.pivot(index=["SegmentDir", "Month"], columns="AccidentSeverity", values="Count").reset_index()
monthlyAccidentSegment.columns = ["SegmentDir", "Month", "Fatal", "Serious", "Slight"]
monthlyAccidentSegment[["Fatal", "Serious", "Slight"]] = monthlyAccidentSegment[["Fatal", "Serious", "Slight"]].fillna(0)

monthlyAccidentSegment.to_csv("MonthlyAccidentSegments.csv", index=False)


