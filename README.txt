THIS IS A PAST TRAINING COURSE EXERCISE THAT HAS BEEN COPIED OVER TO MY PERSONAL GITHUB.

This repository contains a review of England's Smart Motorway Programme using publicly available traffic data.

Please view the PowerBI workbook titled EngSMOverview.pbix for a narrative overview of this research and results.

The various folders are as follows:
- Accident Data: The raw STATS19 accident records.
- Manual Segment Identification: Identified road segments exported from GIS, each manually assigned to the most appropriate MIDAS (traffic count) site.
- Output Images: Images produced through GIS.
- Processed Data: Data that has been wrangled in Python and exported in excel format for use in Power BI.
- Python Analysis: Several python scripts (described individually) performing a number of data formatting tasks, and the main analysis script. This includes importing flow data through the WebTRIS API, cleaning, aggregating and initial visualisations. STATS19 accident data is also formatted and aggregated through a combination of Python scripts here and directly in GIS.
- shps: The shapefiles used in the GIS work.

