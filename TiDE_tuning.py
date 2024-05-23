import random

from darts.dataprocessing.transformers import StaticCovariatesTransformer, MissingValuesFiller
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.pipeline import Pipeline
from darts.models import TiDEModel
from darts import TimeSeries
from tqdm import tqdm
import pandas as pd
import numpy as np
import utils
import json

TIME_COL = "Date"
TARGET = "Weekly_Sales"
RES_TARGET = "residuals"
STATIC_COV = ["Store", "Dept", "Type", "Size"]
DYNAMIC_COV_FILL_0 = ["IsHoliday", 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
DYNAMIC_COV_FILL_INTERPOLATE = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
FREQ = "W-FRI"

SCALER = Scaler()
TRANSFORMER = StaticCovariatesTransformer()
PIPELINE = Pipeline([SCALER, TRANSFORMER])

FORECAST_HORIZON = 10  # number of weeks to forecast
TOP_STORES = 500  # number of top stores to forecast

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

top_stores = df.groupby(['unique_id']).agg({TARGET: 'sum'}).reset_index().sort_values(by=TARGET, ascending=False).head(
    TOP_STORES)
df = df[df['unique_id'].isin(top_stores['unique_id'])]

################################################## FORECASTING ##################################################


window1_start = pd.to_datetime('2012-01-20')
window1 = (window1_start, window1_start + pd.Timedelta(weeks=FORECAST_HORIZON))
test1 = df[(df[TIME_COL] <= window1[1])]

window2_start = pd.to_datetime('2012-03-16')
window2 = (window2_start, window2_start + pd.Timedelta(weeks=FORECAST_HORIZON))
test2 = df[(df[TIME_COL] <= window2[1])]

window3_start = pd.to_datetime('2012-05-25')
window3 = (window3_start, window3_start + pd.Timedelta(weeks=FORECAST_HORIZON))
test3 = df[(df[TIME_COL] <= window3[1])]

window4_start = pd.to_datetime('2012-08-03')
window4 = (window4_start, window4_start + pd.Timedelta(weeks=FORECAST_HORIZON))
test4 = df[(df[TIME_COL] <= window4[1])]

tests = [test1, test2, test3, test4]
windows = [window1, window2, window3, window4]


def create_train_model(params, window):
    chronos_forecast = pd.read_csv('data/chronos_forecast_2012-01-20_2012-03-30.csv')
    chronos_forecast['Date'] = pd.to_datetime(chronos_forecast['Date'])

    residuals = pd.read_csv('data/residuals.csv')
    residuals[TIME_COL] = pd.to_datetime(residuals[TIME_COL])
    residuals[['Store', 'Dept']] = residuals['unique_id'].str.split('-', expand=True).astype(int)

    residuals_train = residuals[residuals[TIME_COL] <= window[0]]
    residuals = residuals[(residuals[TIME_COL] <= window[1])]

    residuals_darts = TimeSeries.from_group_dataframe(
        df=residuals_train,
        group_cols=STATIC_COV,
        time_col=TIME_COL,
        value_cols=RES_TARGET,
        freq=FREQ,
        fill_missing_dates=True,
        fillna_value=0)

    print(
        f"Weeks for training: {len(residuals_train[TIME_COL].unique())} from {min(residuals_train[TIME_COL]).date()} to {max(residuals_train[TIME_COL]).date()}")

    # create dynamic covariates for each serie in the training darts
    dynamic_covariates = []
    for serie in residuals_darts:
        # add the month and week as a covariate
        covariate = datetime_attribute_timeseries(
            serie,
            attribute="month",
            one_hot=True,
            cyclic=False,
            add_length=FORECAST_HORIZON,
        )
        covariate = covariate.stack(
            datetime_attribute_timeseries(
                serie,
                attribute="week",
                one_hot=True,
                cyclic=False,
                add_length=FORECAST_HORIZON,
            )
        )

        store = serie.static_covariates['Store'].item()
        dept = serie.static_covariates['Dept'].item()

        # create covariates to fill with 0
        covariate = covariate.stack(
            TimeSeries.from_dataframe(residuals[(residuals['Store'] == store) & (residuals['Dept'] == dept)],
                                      time_col=TIME_COL, value_cols=DYNAMIC_COV_FILL_0, freq=FREQ,
                                      fill_missing_dates=True,
                                      fillna_value=0)
        )

        # create covariates to fill with interpolation
        dyn_cov_interp = TimeSeries.from_dataframe(
            residuals[(residuals['Store'] == store) & (residuals['Dept'] == dept)],
            time_col=TIME_COL, value_cols=DYNAMIC_COV_FILL_INTERPOLATE, freq=FREQ,
            fill_missing_dates=True)
        covariate = covariate.stack(MissingValuesFiller().transform(dyn_cov_interp))

        dynamic_covariates.append(covariate)

    # scale covariates
    dynamic_covariates_transformed = SCALER.fit_transform(dynamic_covariates)

    # scale data and transform static covariates
    data_transformed = PIPELINE.fit_transform(residuals_darts)

    model = TiDEModel(**params)
    model.fit(data_transformed, future_covariates=dynamic_covariates_transformed, verbose=False)
    pred = PIPELINE.inverse_transform(
        model.predict(n=FORECAST_HORIZON, series=data_transformed, future_covariates=dynamic_covariates_transformed,
                      num_samples=50))
    residuals_forecast = utils.transform_predictions_to_pandas(pred, RES_TARGET, residuals_darts, [0.25, 0.5, 0.75],
                                                               convert=False)

    combined_df = pd.concat([chronos_forecast, residuals_forecast])

    return combined_df.groupby(['unique_id', 'Date']).agg({
        'forecast_lower': 'sum',
        'forecast': 'sum',
        'forecast_upper': 'sum'
    }).reset_index()


# Define the original parameters

with open('best_tested_params.json', 'r') as file:
    best_params = json.load(file)

best_params['input_chunk_length'] = len(residuals_train[TIME_COL].unique()) - FORECAST_HORIZON
best_params['output_chunk_length'] = FORECAST_HORIZON
best_params['likelihood'] = QuantileRegression(quantiles=[0.25, 0.5, 0.75])

# Define the grid of parameters to explore
param_grid = {
    "input_chunk_length": random.choice([4, 8, 16, 26, 52]),
    "num_encoder_layers": random.choice([1, 2, 4, 8]),
    "num_decoder_layers": random.choice([1, 2, 4, 8]),
    "decoder_output_dim": random.choice([1, 4, 8, 16, 32]),
    "hidden_size": random.choice([2, 4, 8, 16]),
    "temporal_width_past": random.choice([2, 4, 8]),
    "temporal_width_future": random.choice([2, 4, 8]),
    "temporal_decoder_hidden": random.choice([4, 8, 16, 26]),
    "dropout": random.choice([0.1, 0.15, 0.3]),
    "batch_size": random.choice([8, 16, 32]),
    "n_epochs": random.choice([15, 10, 25]),
    "random_state": 42,
    "use_static_covariates": True,
    "optimizer_kwargs": {"lr": random.choice([0.001, 0.0001, 0.00001])},
    "use_reversible_instance_norm": random.choice([True, False]),
}

# Initialize the lowest MAPE and best parameters
lowest_mape = 1


# Function to train and evaluate model
def train_and_evaluate(params, window, test):
    # Your training and evaluation logic here
    forecast = create_train_model(params, window)
    t_m = utils.evaluation_metrics(forecast, test)
    return t_m


# Iterate over each parameter in the grid
total_iterations = sum(len(values) for values in param_grid.values())
progress = tqdm(total=total_iterations, desc='Processing', unit='it')

print("Starting hyperparameter tuning:1 \n\n")
for param_key, values in param_grid.items():
    for value in values:
        # Update only the current parameter being tested
        current_params = best_params.copy()
        current_params[param_key] = value

        temp_mape = train_and_evaluate(current_params)

        if temp_mape < lowest_mape:
            lowest_mape = temp_mape
            best_params[param_key] = value

        progress.update(1)
        print(f"MAPE: {temp_mape} - Best MAPE: {lowest_mape}\n\n\n")

progress.close()

# remove object from best_params
best_params.pop('input_chunk_length')
best_params.pop('output_chunk_length')
best_params.pop('likelihood')

# Save the best forecast parameters to a file
with open('params.json', 'w') as file:
    json.dump(best_params, file, indent=4)

print("Best hyperparameters found and saved to file.")
