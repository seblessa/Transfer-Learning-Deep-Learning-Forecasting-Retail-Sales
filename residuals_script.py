from chronos import ChronosPipeline
from darts import TimeSeries
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess
import utils
import torch

subprocess.check_call(['pip3', 'install', '-r', 'requirements.txt'])

TIME_COL = "Date"
TARGET = "Weekly_Sales"
STATIC_COV = ["Store", "Dept", "Type", "Size"]
FREQ = "W-FRI"

################################################## LOADING DATA ##################################################


# load data and exogenous features
df = pd.read_csv('data/train.csv')
store_info = pd.read_csv('data/stores.csv')
exo_feat = pd.read_csv('data/features.csv').drop(columns='IsHoliday')

# join all data frames
df = pd.merge(df, store_info, on=['Store'], how='left')
df = pd.merge(df, exo_feat, on=['Store', TIME_COL], how='left')

# create unique id
df["unique_id"] = df['Store'].astype(str) + '-' + df['Dept'].astype(str)

################################################## PRE-PROCESS DATA ##################################################


df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df[TARGET] = np.where(df[TARGET] < 0, 0, df[TARGET])  # remove negative values
df[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = df[
    ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)  # fill missing values with nan
df["IsHoliday"] = df["IsHoliday"] * 1  # convert boolean into binary
df["Size"] = np.where(df["Size"] < store_info["Size"].quantile(0.25), "small",
                      np.where(df["Size"] > store_info["Size"].quantile(0.75), "large",
                               "medium"))  # make size a categorical variable

# reduce running time by forecasting only top 500 stores-depts1
top_500_stores = df.groupby(['unique_id']).agg({TARGET: 'sum'}).reset_index().sort_values(by=TARGET,
                                                                                          ascending=False).head(500)
df = df[df['unique_id'].isin(top_500_stores['unique_id'])]

################################################## FORECASTING ##################################################

FORECAST_HORIZON = 1  # number of weeks to forecast

START_PREDICTIONS = min(df[TIME_COL]) + (max(df[TIME_COL]) - min(df[TIME_COL])) / 2

# Calculate the difference between the end and start dates
NUM_WEEKS = (max(df[TIME_COL]) - START_PREDICTIONS).days // 7  # number of weeks to process

# ARCHITECTURE = ("amazon/chronos-t5-tiny", "cpu")
ARCHITECTURE = ("amazon/chronos-t5-large","cuda")

# Load the Chronos pipeline
pipeline = ChronosPipeline.from_pretrained(ARCHITECTURE[0], device_map=ARCHITECTURE[1], torch_dtype=torch.bfloat16)

all_residuals_list = []
progress_bar = tqdm(total=NUM_WEEKS, desc='Progress', position=0)


def process_iteration(i):
    start_pred = START_PREDICTIONS + pd.Timedelta(weeks=i)
    test_end = START_PREDICTIONS + pd.Timedelta(weeks=FORECAST_HORIZON + i)

    train = df[(df[TIME_COL] <= start_pred)]
    test = df[(df[TIME_COL] >= start_pred) & (df[TIME_COL] < test_end)]

    # Read train and test datasets and transform train dataset
    train_darts = TimeSeries.from_group_dataframe(
        df=train,
        group_cols=STATIC_COV,
        time_col=TIME_COL,
        value_cols=TARGET,
        freq=FREQ,
        fill_missing_dates=True,
        fillna_value=0)

    forecast = []
    for ts in train_darts:
        # Forecast
        lower, mid, upper = utils.chronos_forecast(pipeline, ts.pd_dataframe().reset_index(), FORECAST_HORIZON)
        forecast.append(utils.convert_forecast_to_pandas([lower, mid, upper], test[
            test['unique_id'] == str(int(list(ts.static_covariates_values())[0][0])) + '-' + str(
                int(list(ts.static_covariates_values())[0][1]))]))
    # Convert list to data frames
    forecast = pd.concat(forecast)

    residuals = pd.DataFrame(test[["unique_id", "Date", "IsHoliday", "Type", "Size", "Temperature", "Fuel_Price", "CPI",
                                   "Unemployment", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]])
    residuals["residuals"] = test["Weekly_Sales"] - forecast["forecast"]
    last_column = residuals.pop(residuals.columns[-1])
    residuals.insert(2, last_column.name, last_column)

    progress_bar.update(1)
    return residuals


# Use ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_iteration, i) for i in range(NUM_WEEKS)]
    for future in concurrent.futures.as_completed(futures):
        all_residuals_list.append(future.result())

progress_bar.close()

# Concatenate all residuals into a single DataFrame
all_residuals = pd.concat(all_residuals_list, ignore_index=True).sort_values(by=['unique_id', 'Date'])
all_residuals.to_csv('data/residuals.csv', index=False)
