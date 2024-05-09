import pandas as pd
import numpy as np
import subprocess
import utils


subprocess.check_call(['pip3', 'install', '-r', 'requirements.txt'])

TIME_COL = "Date"
TARGET = "Weekly_Sales"
RES_TARGET = "residuals"

FORECAST_HORIZON = 17  # number of weeks to forecast
TOP_STORES = 500  # number of top stores to forecast

CHRONOS_ARCHITECTURE = ("amazon/chronos-t5-large","cuda")


################################################## LOADING DATA ##################################################


# load data and exogenous features
df = pd.read_csv('data/train.csv')
store_info = pd.read_csv('data/stores.csv')
exo_feat = pd.read_csv('data/features.csv').drop(columns='IsHoliday')

# join all data frames
df = pd.merge(df, store_info, on=['Store'], how='left')
df = pd.merge(df, exo_feat, on=['Store', TIME_COL], how='left')

# create unique id
df["unique_id"] = df['Store'].astype(str)+'-'+df['Dept'].astype(str)

print(f"Distinct number of time series: {len(df['unique_id'].unique())}")

################################################## PRE-PROCESS DATA ##################################################


df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df[TARGET] = np.where(df[TARGET] < 0, 0, df[TARGET]) # remove negative values
df[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5']] = df[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4','MarkDown5']].fillna(0) # fill missing values with nan
df["IsHoliday"] = df["IsHoliday"]*1 # convert boolean into binary
df["Size"] = np.where(df["Size"] < store_info["Size"].quantile(0.25), "small",
                np.where(df["Size"] > store_info["Size"].quantile(0.75), "large",
                "medium")) # make size a categorical variable

top_stores = df.groupby(['unique_id']).agg({TARGET: 'sum'}).reset_index().sort_values(by=TARGET, ascending=False).head(TOP_STORES)
df = df[df['unique_id'].isin(top_stores['unique_id'])]

print(f"Distinct number of time series: {len(df['unique_id'].unique())}")


window1_start=pd.to_datetime('2012-01-20')
window1=(window1_start,window1_start + pd.Timedelta(weeks=FORECAST_HORIZON))

window2_start = pd.to_datetime('2012-03-30')
window2=(window2_start,window2_start + pd.Timedelta(weeks=FORECAST_HORIZON))

window3_start = pd.to_datetime('2012-06-08')
window3=(window3_start,window3_start + pd.Timedelta(weeks=FORECAST_HORIZON))

window4_start = pd.to_datetime('2012-08-17')
window4=(window4_start,window4_start + pd.Timedelta(weeks=FORECAST_HORIZON))

windows = [window1, window2, window3, window4]



################################################## FORECASTING ##################################################

for window in windows:
    utils.chronos_prediction(window, df, CHRONOS_ARCHITECTURE, FORECAST_HORIZON)
    utils.tide_prediction(window, df, FORECAST_HORIZON)
