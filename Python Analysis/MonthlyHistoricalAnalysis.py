# Contains the main body of the analysis; produces plots and includes some attempts at modelling.
# The code used in the PowerBI workbook is copied from here to allow for dyanmic plots.

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import scipy.stats as stats

# Fix issue with PyCharm where all plot windows crash when using new MatPlotLib backend by reverting to older backend.
matplotlib.use("Qt5Agg")

#### IMPORT, CLEAN, AND FORMAT DATA
flowDataPath = "SpeedDataImport_7days.csv"
monthlyData = pd.read_csv(flowDataPath)
accidentDataPath = "MonthlyAccidentSegments.csv"
accidents = pd.read_csv(accidentDataPath)

accidents["Month"] = pd.to_datetime(accidents["Month"], format="%Y-%m-%d")
accidents["TotalAccidents"] = accidents[["Slight", "Serious", "Fatal"]].sum(axis=1)

monthlyData["StartDate"] = pd.to_datetime(monthlyData["StartDate"], format="%d/%m/%Y")
monthlyData["Month"] = pd.to_datetime(monthlyData["Month"], format="%Y-%m-%d")
monthlyData["MonthStart"] = monthlyData["StartDate"].to_numpy().astype("datetime64[M]")

# Remove records where ADT or AverageSpeed is 0. Could be legitimate road closure or recording error.
monthlyData = monthlyData.loc[(monthlyData["ADT"] != 0) & (monthlyData["AverageSpeed"] != 0)]
# Remove records from the month in which SM is activated to avoid mixed pre- and post- records.
monthlyData = monthlyData.loc[monthlyData["Month"] != monthlyData["MonthStart"]]
# Flag whether a record corresponds to an active SM scheme or not.
monthlyData["ActiveType"] = np.where(monthlyData["Month"] > monthlyData["MonthStart"], monthlyData["Type"], "No SM")

# Estimate monthly vehicle km from ADT to allow us to compare months which don't all have complete data.
monthlyData["DaysInMonth"] = monthlyData["Month"].dt.daysinmonth
monthlyData["MonthVehKM"] = monthlyData["ADT"] * monthlyData["DaysInMonth"] * (monthlyData["SegLength"] / 1000)

# Join accident data.
monthlyData = monthlyData.merge(accidents,
                    how="left",
                    on=["SegmentDir", "Month"])
monthlyData[["Fatal", "Serious", "Slight", "TotalAccidents"]] = monthlyData[
    ["Fatal", "Serious", "Slight", "TotalAccidents"]].fillna(0)

# COVID FILTER: Remove all Covid-19 pandemic data.
monthlyData = monthlyData.loc[(monthlyData["Month"] < "2020-03-01") | (monthlyData["Month"] > "2021-12-31")]

#### MONTHLY DATA ANALYSIS
# Choose plotting colours
colourLookup = {"No SM": "#424B54",
                "Controlled": "#F2542D",
                "Dynamic": "#FFCF00",
                "ALR": "#0E9594"}
colourOrder = ["No SM", "Controlled", "Dynamic", "ALR"]

# Scatter chart of monthly ADT vs Avg Speed.
speedFlowPlot = sns.relplot(data=monthlyData, x="ADT", y="AverageSpeed", hue="ActiveType", col="ActiveType", col_wrap=4,
                            palette=colourLookup, col_order=colourOrder, hue_order=colourOrder, kind="scatter",
                            height=5, aspect=1)
speedFlowPlot.fig.suptitle("Speed-Flow Relationship by SM Type", fontsize=16)
speedFlowPlot.set_axis_labels("ADT", "Average Speed (mph)")
speedFlowPlot.set_titles("")
plt.subplots_adjust(top=0.8)
sns.move_legend(
    speedFlowPlot, "center",
    bbox_to_anchor=(.5, 0.88), ncol=4, title=None, frameon=False,
)
plt.setp(speedFlowPlot._legend.get_texts(), fontsize=14)
speedFlowPlot.fig.show()

# After several iterations of data cleaning and formatting there are no more outliers that are believed to be caused by data errors.


#### AGGREGATE DATA ANALYSIS
# Calculate accidents per billion veh km over the whole lifespan of each segment / SM-type pair.
# Custom weighted average aggregation function to get more accurate average speed over each period.
def volume_weighted_speed(speed, volume):
    return (speed * volume).sum() / volume.sum()


aggData = monthlyData.loc[:, ["SegmentDir", "ActiveType", "MonthVehKM", "TotalAccidents", "AverageSpeed", "ADT"]]
aggData = aggData.groupby(["SegmentDir", "ActiveType"]).agg({"MonthVehKM": "sum",
                                                                         "TotalAccidents": "sum",
                                                                         "ADT": "mean",
                                                                         "AverageSpeed": lambda
                                                                             x: volume_weighted_speed(x,
                                                                                                      aggData.loc[
                                                                                                          x.index, "MonthVehKM"])}).reset_index()
aggData = aggData.rename(columns={"MonthVehKM": "TotalVehKM"})
aggData["AccidentRate"] = aggData["TotalAccidents"] / (aggData["TotalVehKM"] / 1000000000)
aggData = aggData.merge(monthlyData[["SegmentDir", "Type"]].drop_duplicates(), how="left", on="SegmentDir")

# Scatter chart of accident rates.
accidentRateScatter = sns.relplot(data=aggData, x="ADT", y="AccidentRate", hue="ActiveType",
                                  col="ActiveType", col_wrap=4, palette=colourLookup,
                                  col_order=colourOrder, kind="scatter", height=5, aspect=1)
accidentRateScatter.fig.suptitle("Accident Rates by SM Type", fontsize=16)
accidentRateScatter.set_axis_labels("ADT", "Accidents per Billion Vehicle Kilometers")
accidentRateScatter.set_titles("")
plt.subplots_adjust(top=0.8)
sns.move_legend(
    accidentRateScatter, "center",
    bbox_to_anchor=(.5, 0.88), ncol=4, title=None, frameon=False)
plt.setp(accidentRateScatter._legend.get_texts(), fontsize=14)
accidentRateScatter.fig.show()


# Function to add labels to faceted plot.
def plot_count_labels(data, column, **kwargs):
    n = data[column].count()
    ax = plt.gca()
    plt.text(1, 0.9, f"N = {n}", ha="center", va="center", fontsize=14, transform=ax.transAxes)


# Histogram of accident rates
accidentRateHist = sns.FacetGrid(data=aggData, hue="ActiveType", col="ActiveType", col_wrap=4,
                                 palette=colourLookup, col_order=colourOrder, height=5, aspect=1, xlim=(0, 325))
accidentRateHist.map(sns.histplot, "AccidentRate", binwidth=10)
accidentRateHist.map_dataframe(plot_count_labels, column="AccidentRate")
accidentRateHist.fig.suptitle("Accident Rates by SM Type", fontsize=16)
accidentRateHist.set_axis_labels("Accidents per Billion Vehicle KM", "Frequency")
accidentRateHist.set_titles("")
plt.subplots_adjust(top=0.8)
accidentRateHist.add_legend(labels=colourOrder)
sns.move_legend(
    accidentRateHist, "center",
    bbox_to_anchor=(.5, 0.88), ncol=4, title=None, frameon=False,
)
plt.setp(accidentRateHist._legend.get_texts(), fontsize=14)
plt.gcf().subplots_adjust(bottom=0.14)
accidentRateHist.fig.text(0.27, 0.01,
                          "M6 J11-11a SB is not shown due to extremely high accident rates caused by low traffic volumes. The \"No-SM\" accident rate is 3,167; the \"Controlled\" rate is 931.",
                          ha="center", fontsize=10, fontstyle="italic")
accidentRateHist.fig.show()


#### STATISTICAL SIGNIFICANCE TESTING
# Format data for tests.
aggDataWide = aggData.copy()
aggDataWide["ActiveType"] = np.where(aggData["ActiveType"] == "No SM", aggData["ActiveType"], "SM")
aggDataWide = aggDataWide.pivot(index=["SegmentDir", "Type"],
                                                          columns="ActiveType",
                                                          values=["ADT", "AverageSpeed", "AccidentRate"]).reset_index()

# Exclude rows where we don't have both pre- and post-SM values using NaNs.
# Note: This removes all Dynamic sites.
for i in ["AccidentRate", "ADT", "AverageSpeed"]:
    mask = aggDataWide[(i, "No SM")].isnull() | aggDataWide[(i, "SM")].isnull()
    aggDataWide = aggDataWide.drop(aggDataWide[mask].index)

# Calculate percentage changes for plotting later.
aggDataWide["PercChangeADT"] = (aggDataWide["ADT"]["SM"] - aggDataWide["ADT"]["No SM"]) / aggDataWide["ADT"]["No SM"]
aggDataWide["PercChangeSpeed"] = (aggDataWide["AverageSpeed"]["SM"] - aggDataWide["AverageSpeed"]["No SM"]) / aggDataWide["AverageSpeed"]["No SM"]
aggDataWide["PercChangeAccidentRate"] = (aggDataWide["AccidentRate"]["SM"] - aggDataWide["AccidentRate"]["No SM"]) / aggDataWide["AccidentRate"]["No SM"]
# Raw change also considered for Accident Rate due to presence of zeros in No SM data.
aggDataWide["ChangeAccidentRate"] = aggDataWide["AccidentRate"]["SM"] - aggDataWide["AccidentRate"]["No SM"]
aggDataWide = aggDataWide.replace(np.inf, np.nan)

# Paired t-tests: ALR
aggDataWide_ALR = aggDataWide.loc[aggDataWide["Type"] == "ALR"].copy()
# Accident Rate: t-statistic = 3.88, p-value = 0.00042
ALR_AR_tt = stats.ttest_rel(a=aggDataWide_ALR["AccidentRate"]["No SM"], b=aggDataWide_ALR["AccidentRate"]["SM"], nan_policy="omit")
# ADT: t-statistic = -4.16, p-value = 0.00019
ALR_ADT_tt = stats.ttest_rel(a=aggDataWide_ALR["ADT"]["No SM"], b=aggDataWide_ALR["ADT"]["SM"], nan_policy='omit')
# Average Speed: t-statistic = -4.02, p-value = 0.00028
ALR_Speed_tt = stats.ttest_rel(a=aggDataWide_ALR["AverageSpeed"]["No SM"], b=aggDataWide_ALR["AverageSpeed"]["SM"], nan_policy='omit')

# Paired t-tests: Controlled Motorway
aggDataWide_Controlled = aggDataWide.loc[aggDataWide["Type"] == "Controlled"]
# Accident Rate: t-statistic = -0.08, p-value = 0.93
Controlled_AR_tt = stats.ttest_rel(a=aggDataWide_Controlled["AccidentRate"]["No SM"], b=aggDataWide_Controlled["AccidentRate"]["SM"], nan_policy="omit")
# ADT: t-statistic = -0.92, p-value = 0.39
Controlled_ADT_tt = stats.ttest_rel(a=aggDataWide_Controlled["ADT"]["No SM"], b=aggDataWide_Controlled["ADT"]["SM"], nan_policy='omit')
# Average Speed: t-statistic = -5.29, p-value = 0.0011
Controlled_Speed_tt = stats.ttest_rel(a=aggDataWide_Controlled["AverageSpeed"]["No SM"], b=aggDataWide_Controlled["AverageSpeed"]["SM"], nan_policy='omit')

# Manually construct significance test results dataframe for export.
tTestExport = pd.DataFrame({"Metric": ["Accident Rate", "ADT", "Average Speed"],
                            "ALR": [ALR_AR_tt[1], ALR_ADT_tt[1], ALR_Speed_tt[1]],
                            "Controlled": [Controlled_AR_tt[1], Controlled_ADT_tt[1], Controlled_Speed_tt[1]]})

#### DIFFERENCE EXAMINATION
# Histogram ADT percentage change.
colourLookup_Change = {"Controlled": "#F2542D",
                       "ALR": "#0E9594"}
colourOrder_Change = ["Controlled", "ALR"]
ADT_ChangePlot = sns.FacetGrid(data=aggDataWide, hue=("Type", ""), col=("Type", ""), col_wrap=2, xlim=(-1, 1),
                               palette=colourLookup_Change, col_order=colourOrder_Change, height=5, aspect=1.1)
ADT_ChangePlot.map(sns.histplot, ("PercChangeADT", ""), bins=np.linspace(-1, 1, 21))
ADT_ChangePlot.map_dataframe(plot_count_labels, column=("PercChangeADT", ""))
ADT_ChangePlot.fig.suptitle("Percentage Change in ADT After Smart Motorway Implementation", fontsize=16)
ADT_ChangePlot.set_axis_labels("ADT Change (%)", "Number of Scheme Sites")
ADT_ChangePlot.set_titles("")
ADT_ChangePlot.add_legend(labels=colourOrder_Change)
sns.move_legend(ADT_ChangePlot, "center", bbox_to_anchor=(.5, 0.88), ncol=4, title=None, frameon=False)
for ax in ADT_ChangePlot.axes:
    ax.set_xticks(ticks=np.linspace(-1, 1, 11))
    ax.set_yticks(ticks=np.linspace(0, 20, 6))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.axvline(x=0, color='k', linestyle='--', linewidth=2.5)
plt.subplots_adjust(top=0.8)
plt.setp(ADT_ChangePlot._legend.get_texts(), fontsize=14)
ADT_ChangePlot.fig.show()

# Histogram Average Speed percentage change.
Speed_ChangePlot = sns.FacetGrid(data=aggDataWide, hue=("Type", ""), col=("Type", ""), col_wrap=2, xlim=(-1, 1),
                                 palette=colourLookup_Change, col_order=colourOrder_Change, height=5, aspect=1.1)
Speed_ChangePlot.map(sns.histplot, ("PercChangeSpeed", ""), bins=np.linspace(-1, 1, 21))
Speed_ChangePlot.map_dataframe(plot_count_labels, column=("PercChangeSpeed", ""))
Speed_ChangePlot.fig.suptitle("Percentage Change in Average Speed After Smart Motorway Implementation", fontsize=16)
Speed_ChangePlot.set_axis_labels("Average Speed Change (%)", "Number of Scheme Sites")
Speed_ChangePlot.set_titles("")
Speed_ChangePlot.add_legend(labels=colourOrder_Change)
sns.move_legend(Speed_ChangePlot, "center", bbox_to_anchor=(.5, 0.88), ncol=4, title=None, frameon=False)
for ax in Speed_ChangePlot.axes:
    ax.set_xticks(ticks=np.linspace(-1, 1, 11))
    ax.set_yticks(ticks=np.linspace(0, 20, 6))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.axvline(x=0, color='k', linestyle='--', linewidth=2.5)
plt.subplots_adjust(top=0.8)
plt.setp(Speed_ChangePlot._legend.get_texts(), fontsize=14)
Speed_ChangePlot.fig.show()

# Histogram Accident Rate percentage change.
Accident_ChangePlot = sns.FacetGrid(data=aggDataWide, hue=("Type", ""), col=("Type", ""), col_wrap=2, xlim=(-1, 1),
                                    palette=colourLookup_Change, col_order=colourOrder_Change, height=5, aspect=1.1)
Accident_ChangePlot.map(sns.histplot, ("PercChangeAccidentRate", ""), bins=np.linspace(-2.5, 2.5, 51))
Accident_ChangePlot.map_dataframe(plot_count_labels, column=("PercChangeAccidentRate", ""))
Accident_ChangePlot.fig.suptitle("Percentage Change in Accident Rate After Smart Motorway Implementation", fontsize=16)
Accident_ChangePlot.set_axis_labels("Change in Accidents per Billion Veh-km (%)", "Number of Scheme Sites")
Accident_ChangePlot.set_titles("")
Accident_ChangePlot.add_legend(labels=colourOrder_Change)
sns.move_legend(Accident_ChangePlot, "center", bbox_to_anchor=(.5, 0.88), ncol=4, title=None, frameon=False)
for ax in Accident_ChangePlot.axes:
    ax.set_xticks(ticks=np.linspace(-2.5, 2.5, 11))
    ax.set_yticks(ticks=np.linspace(0, 20, 6))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.axvline(x=0, color='k', linestyle='--', linewidth=2.5)
plt.subplots_adjust(top=0.8)
plt.setp(Accident_ChangePlot._legend.get_texts(), fontsize=14)
Accident_ChangePlot.fig.show()

# Correlation
# No SM Correlation
corrNoSM = aggData.loc[aggData["ActiveType"] == "No SM", ["ADT", "AverageSpeed", "AccidentRate"]].corr()
fig, ax1 = plt.subplots(figsize=(5, 5))
ax1.matshow(corrNoSM, cmap='RdYlGn')
ax1.set_xticks(np.linspace(0, 2, 3))
ax1.set_yticks(np.linspace(0, 2, 3))
ax1.set_xticklabels(corrNoSM.index)
ax1.set_yticklabels(corrNoSM.columns)
ax1.set_title('Correlation: No Smart Motorway Scheme')
for (i, j), z in np.ndenumerate(corrNoSM):
    ax1.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

# ALR Correlation
corrALR = aggData.loc[aggData["ActiveType"] == "ALR", ["ADT", "AverageSpeed", "AccidentRate"]].corr()
fig, ax1 = plt.subplots(figsize=(5, 5))
ax1.matshow(corrALR, cmap='RdYlGn')
ax1.set_xticks(np.linspace(0, 2, 3))
ax1.set_yticks(np.linspace(0, 2, 3))
ax1.set_xticklabels(corrALR.index)
ax1.set_yticklabels(corrALR.columns)
ax1.set_title('Correlation: ALR Smart Motorway')
for (i, j), z in np.ndenumerate(corrALR):
    ax1.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

# Controlled Correlation
corrControlled = aggData.loc[aggData["ActiveType"] == "Controlled", ["ADT", "AverageSpeed", "AccidentRate"]].corr()
fig, ax1 = plt.subplots(figsize=(5, 5))
ax1.matshow(corrControlled, cmap='RdYlGn')
ax1.set_xticks(np.linspace(0, 2, 3))
ax1.set_yticks(np.linspace(0, 2, 3))
ax1.set_xticklabels(corrControlled.index)
ax1.set_yticklabels(corrControlled.columns)
ax1.set_title('Correlation: Controlled Smart Motorway')
for (i, j), z in np.ndenumerate(corrControlled):
    ax1.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

# Export some data for PowerBI
# Removes multi-level column labels and pivot to long data for use with PowerBI filtering.
aggDataWideExport = aggDataWide.copy()
aggDataWideExport.columns = ['_'.join(col).rstrip('_') if col[1] else col[0] for col in aggDataWideExport.columns.values]
aggDataWideExport = aggDataWideExport[["SegmentDir", "Type", "PercChangeADT", "PercChangeSpeed", "PercChangeAccidentRate"]]
aggDataWideExport = aggDataWideExport.rename(columns={"PercChangeADT": "ADT",
                                                      "PercChangeSpeed": "Average Speed",
                                                      "PercChangeAccidentRate": "Accident Rate"})
aggDataLong = pd.melt(aggDataWideExport, id_vars=["SegmentDir", "Type"], value_vars=["ADT", "Average Speed", "Accident Rate"], var_name="Metric", value_name="Value")
aggDataLong = aggDataLong.rename(columns={"variable": "Metric", "value": "Value"})

outPath = "../Processed Data/"
monthlyData.to_csv(outPath+"monthlyData.csv", index=False)
aggDataLong.to_csv(outPath+"aggDataLong.csv", index=False)
tTestExport.to_csv(outPath+"tTest.csv", index=False)

## TESTING SOME MODELLING APPROACHES
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

# Splits for KFold validation.
n_splits = 5

# LINEAR MODELLING
df = aggData.loc[aggData["Type"] == "ALR"].copy()
df["SMFlag"] = np.where(df["ActiveType"] == "No SM", 0, 1)

X = df[["ADT", "AverageSpeed", "SMFlag"]].reset_index(drop=True)
y = df["AccidentRate"].reset_index(drop=True)

# Standardize features.
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

test_scores = []
# Loop over each fold
for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit a linear regression model to the training data
    lin = linear_model.LinearRegression()
    lin.fit(X_train, y_train)

    # Print score
    y_pred = lin.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    test_scores.append(mae)

    print(lin.coef_)

mean_score = np.mean(test_scores)
print("Linear Negative Mean Abs Error: {:.3f}".format(-mean_score))

# MAE: -25.75

# LINEAR MODEL, FIT OVER WHOLE DATASET AND CHECK SIGNIFICANCE OF COEFFS
linAll = linear_model.LinearRegression()
linAll.fit(X, y)
y_pred = linAll.predict(X)
mae = mean_absolute_error(y, y_pred)
coef = linAll.coef_

# calculate the p-values for each coefficient
n_samples = len(y)
n_features = X.shape[1]
dof = n_samples - n_features - 1
alpha = 0.05
resid = y - y_pred
sse = np.sum(resid ** 2)
se = np.sqrt(sse / dof)
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
X_norm_T = X_norm.T
X_norm_T_X_norm = X_norm_T.dot(X_norm)
cov = np.linalg.inv(X_norm_T_X_norm) * (se ** 2)
std_err = np.sqrt(np.diag(cov))

# calculate the t-statistics for each coefficient
t_stats = coef / std_err

# calculate the p-values for each coefficient
p_values = stats.t.sf(np.abs(t_stats), dof) * 2

feature_names = ["ADT", "Average Speed", "Smart Motorway (Binary)"]
# Store coeffs in df
data = {'Coefficient': coef, 'P-value': p_values}
coef_significance = pd.DataFrame(data, index=feature_names)

coef_significance.to_csv(outPath+"linearModelCoefficients.csv")

# POLYNOMIAL MODELLING
degree = 2

test_scores = []
# Loop over each fold
for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets for this fold
    X_train_poly, X_test_poly = X[train_index], X[test_index]
    y_train_poly, y_test_poly = y[train_index], y[test_index]

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_poly)
    X_test_poly = poly.fit_transform(X_test_poly)

    # Fit a linear regression model to the training data
    poly_reg = linear_model.LinearRegression()
    poly_reg.fit(X_train_poly, y_train_poly)

    # Print score
    y_pred_poly = poly_reg.predict(X_test_poly)
    mae = mean_absolute_error(y_test_poly, y_pred_poly)
    test_scores.append(mae)

mean_score = np.mean(test_scores)
print("Poly Negative Mean Abs Error: {:.3f}".format(-mean_score))

# Degree 2: -24.854
# Degree 3: -28.734
# Degree 4: -86.709
# Degree 5: -354.789

# DECISION TREE
test_scores = []

param_grid = {'max_depth': [2, 3, 4, 5, 10, 20, 50, 100, None],
              'min_samples_split': [2, 4, 6, 8],
              'min_samples_leaf': [1, 2, 4, 6]}

# Define the DecisionTreeRegressor model
dt = DecisionTreeRegressor()

# Define the GridSearchCV object
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=kf, n_jobs=-1, scoring="neg_mean_absolute_error")

# Fit the GridSearchCV object to the data
grid_search.fit(X, y)

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)