from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.pipeline import Pipeline
from darts.models import TiDEModel, TSMixerModel
from darts import TimeSeries
import pandas as pd
import numpy as np
import argparse
import random
import utils

TIME_COL = "Date"
TARGET = "Weekly_Sales"
RES_TARGET = "residuals"
STATIC_COV = ["Store", "Dept", "Type", "Size", "unique_id"]
DYNAMIC_COV_FILL_0 = ["IsHoliday", 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
DYNAMIC_COV_FILL_INTERPOLATE = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
FREQ = "W-FRI"

SCALER = Scaler()
TRANSFORMER = StaticCovariatesTransformer()
PIPELINE = Pipeline([SCALER, TRANSFORMER])

FORECAST_HORIZON = 10  # number of weeks to forecast
TOP_STORES = 500  # number of top stores to forecast


def get_df():
    # load data and exogenous features
    df = pd.read_csv('data/train.csv')
    store_info = pd.read_csv('data/stores.csv')
    exo_feat = pd.read_csv('data/features.csv').drop(columns='IsHoliday')

    # join all data frames
    df = pd.merge(df, store_info, on=['Store'], how='left')
    df = pd.merge(df, exo_feat, on=['Store', TIME_COL], how='left')

    # create unique id
    df["unique_id"] = df['Store'].astype(str) + '-' + df['Dept'].astype(str)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df.loc[:, TIME_COL] = pd.to_datetime(df[TIME_COL])

    df[TARGET] = np.where(df[TARGET] < 0, 0, df[TARGET])
    df[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = df[
        ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)  # fill missing values with nan
    df["IsHoliday"] = df["IsHoliday"] * 1  # convert boolean into binary
    df["Size"] = np.where(df["Size"] < store_info["Size"].quantile(0.25), "small",
                          np.where(df["Size"] > store_info["Size"].quantile(0.75), "large",
                                   "medium"))  # make size a categorical variable

    top_stores = df.groupby(['unique_id']).agg({TARGET: 'sum'}).reset_index().sort_values(by=TARGET,
                                                                                          ascending=False).head(
        TOP_STORES)

    return df[df['unique_id'].isin(top_stores['unique_id'])]


def create_train_model_residuals(params, window, chronos_forecast, model_creator):
    chronos_forecast[TIME_COL] = pd.to_datetime(chronos_forecast[TIME_COL])

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

    # create dynamic covariates for each serie in the training darts
    dynamic_covariates = utils.create_dynamic_covariates(residuals_darts, residuals, FORECAST_HORIZON)

    # scale covariates
    dynamic_covariates_transformed = SCALER.fit_transform(dynamic_covariates)

    # scale data and transform static covariates
    data_transformed = PIPELINE.fit_transform(residuals_darts)

    model = model_creator(**params)
    model.fit(data_transformed, future_covariates=dynamic_covariates_transformed, verbose=False)
    pred = PIPELINE.inverse_transform(
        model.predict(n=FORECAST_HORIZON, series=data_transformed, future_covariates=dynamic_covariates_transformed,
                      num_samples=50))
    residuals_forecast = utils.transform_predictions_to_pandas(pred, RES_TARGET, residuals_darts, [0.25, 0.5, 0.75],
                                                               convert=False)

    combined_df = pd.concat([chronos_forecast, residuals_forecast])

    return combined_df.groupby(['unique_id', TIME_COL]).agg({
        'forecast_lower': 'sum',
        'forecast': 'sum',
        'forecast_upper': 'sum'
    }).reset_index()


def create_train_model_dyn_covs(df, params, window, chronos_forecast, model_creator):
    chronos_forecast[TIME_COL] = pd.to_datetime(chronos_forecast[TIME_COL])

    train = df[(df[TIME_COL] <= window[0]) & (df[TIME_COL] >= pd.to_datetime('2011-06-17'))]
    df = df[(df[TIME_COL] <= window[1]) & (df[TIME_COL] >= pd.to_datetime('2011-06-17'))]

    residuals = pd.read_csv('data/residuals.csv')
    residuals[TIME_COL] = pd.to_datetime(residuals[TIME_COL])
    df = pd.merge(df, residuals[["unique_id", TIME_COL, "residuals"]], on=["unique_id", TIME_COL], how="left")
    df["forecast"] = df[TARGET] - df["residuals"]

    df = pd.merge(df, chronos_forecast[["unique_id", TIME_COL, "forecast"]], on=["unique_id", TIME_COL], how="left")
    df["forecast"] = np.where(df[TIME_COL] > window[0], df["forecast_y"], df["forecast_x"])

    train_darts = TimeSeries.from_group_dataframe(
        df=train,
        group_cols=STATIC_COV,
        time_col=TIME_COL,
        value_cols=TARGET,
        freq=FREQ,
        fill_missing_dates=True,
        fillna_value=0)

    dynamic_covariates = utils.create_dynamic_covariates(train_darts, df, FORECAST_HORIZON, w_pred=True)

    # scale covariates
    dynamic_covariates_transformed = SCALER.fit_transform(dynamic_covariates)

    # scale data and transform static covariates
    data_transformed = PIPELINE.fit_transform(train_darts)

    model = model_creator(**params)
    model.fit(data_transformed, future_covariates=dynamic_covariates_transformed, verbose=False)
    pred = PIPELINE.inverse_transform(
        model.predict(n=FORECAST_HORIZON, series=data_transformed, future_covariates=dynamic_covariates_transformed,
                      num_samples=50))
    return utils.transform_predictions_to_pandas(pred, TARGET, train_darts, [0.25, 0.5, 0.75])


def save_dict_to_txt(rmse, data_dict, file_name):
    with open(file_name, 'w') as f:
        f.write(f'RMSE: {rmse}\n\n\n\n')
        for key, v in data_dict.items():
            # Convert value to a string if it's not already
            value_str = str(v)
            f.write(f'{key}: {value_str}\n')

    print(f"\nBest params has been saved to '{file_name}'.\n")


# Function to train and evaluate model
def train_and_evaluate(df, params, window, test, chronos, model_creator, approach='residuals'):
    # Your training and evaluation logic here
    if approach == 'residuals':
        forecast = create_train_model_residuals(params, window, chronos, model_creator)
    elif approach == 'dynamic_covariates':
        forecast = create_train_model_dyn_covs(df, params, window, chronos, model_creator)
    else:
        raise ValueError(f"Approach '{approach}' is not valid.")

    return utils.rmse_evaluation(forecast, test)


def find_best_params(df, wds, ts, chr_forecasts, model_creator, param_func, model_name, approach):
    # Initialize the lowest RMSE and best parameters
    lowest_rmse = float('inf')
    start_timestamp = pd.Timestamp.now()

    # Iterate over each parameter in the grid
    print("\n\nStarting hyperparameter tuning: \n\n")

    while True:
        params = param_func()
        rmse = 0
        for window, test, chronos in zip(wds, ts, chr_forecasts):
            rmse += train_and_evaluate(df, params, window, test, chronos, model_creator, approach=approach)
        rmse /= len(wds)

        if rmse < lowest_rmse:
            lowest_rmse = rmse
            save_dict_to_txt(rmse, params, f'HPTuning/found_params_{model_name}_{approach}.txt')

        print(f"\n\n\nRMSE: {rmse} - Best RMSE: {lowest_rmse}")
        print(f"Time elapsed: {pd.Timestamp.now() - start_timestamp}\n\n\n")


def get_tsmixer_params():
    return {
        "input_chunk_length": random.choice([2, 4, 8, 10]),
        "output_chunk_length": FORECAST_HORIZON,
        "hidden_size": random.choice([2, 4, 8, 16]),
        "ff_size": random.choice([2, 4, 8, 16]),
        "num_blocks": random.choice([1, 2, 3, 4]),
        "activation": random.choice(["ELU", "ReLU", "LeakyReLU", "GELU"]),
        "dropout": random.choice([0.1, 0.15, 0.3, 0.35]),
        "normalize_before": random.choice([True, False]),
        "batch_size": random.choice([8, 16, 32, 64]),
        "n_epochs": random.choice([10, 15, 20, 25, 30]),
        "likelihood": QuantileRegression(quantiles=[0.25, 0.5, 0.75]),
        "random_state": 42,
        "use_static_covariates": True,
        "optimizer_kwargs": {"lr": random.choice([1e-3, 1e-4, 1e-5, 1e-6])},
        "use_reversible_instance_norm": random.choice([True, False]),
    }


def get_tide_params():
    return {
        "input_chunk_length": random.choice([2, 3, 4, 6, 7]),
        "output_chunk_length": FORECAST_HORIZON,
        "num_encoder_layers": random.choice([2, 4, 6, 8]),
        "num_decoder_layers": random.choice([2, 4, 6, 8]),
        "decoder_output_dim": random.choice([6, 8, 10, 15, 16]),
        "hidden_size": random.choice([2, 4, 8, 16]),
        "temporal_width_past": random.choice([2, 4, 8]),
        "temporal_width_future": random.choice([4, 8, 10, 12]),
        "temporal_decoder_hidden": random.choice([16, 23, 26, 32]),
        "dropout": random.choice([0.1, 0.15, 0.3]),
        "batch_size": random.choice([8, 16, 32, 64]),
        "n_epochs": random.choice([10, 15, 20, 25, 30]),
        "likelihood": QuantileRegression(quantiles=[0.25, 0.5, 0.75]),
        "random_state": 42,
        "use_static_covariates": True,
        "optimizer_kwargs": {"lr": random.choice([1e-3, 1e-4, 1e-5, 1e-6])},
        "use_reversible_instance_norm": random.choice([True, False]),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to find best hyperparameters for TiDE or TSMixer models.')

    parser.add_argument('-m', '--model', choices=['TiDE', 'TSMixer'], required=True,
                        help="Specify the model to use: 'TiDE' or 'TSMixer'")

    parser.add_argument('-app', '--approach', choices=['residuals', 'dynamic_covariates'], required=True,
                        help="Specify the approach: 'residuals' or 'dynamic_covariates'")
    args = parser.parse_args()

    dataframe = get_df()

    window1_start = pd.to_datetime('2012-01-20')
    window1 = (window1_start, window1_start + pd.Timedelta(weeks=FORECAST_HORIZON))
    test1 = dataframe[(dataframe[TIME_COL] <= window1[1])]

    window2_start = pd.to_datetime('2012-03-16')
    window2 = (window2_start, window2_start + pd.Timedelta(weeks=FORECAST_HORIZON))
    test2 = dataframe[(dataframe[TIME_COL] <= window2[1])]

    window3_start = pd.to_datetime('2012-05-25')
    window3 = (window3_start, window3_start + pd.Timedelta(weeks=FORECAST_HORIZON))
    test3 = dataframe[(dataframe[TIME_COL] <= window3[1])]

    window4_start = pd.to_datetime('2012-08-03')
    window4 = (window4_start, window4_start + pd.Timedelta(weeks=FORECAST_HORIZON))
    test4 = dataframe[(dataframe[TIME_COL] <= window4[1])]

    tests = [test1, test2, test3, test4]
    chronos_forecasts = [
        pd.read_csv('data/chronos_forecast_2012-01-20_2012-03-30.csv'),
        pd.read_csv('data/chronos_forecast_2012-03-16_2012-05-25.csv'),
        pd.read_csv('data/chronos_forecast_2012-05-25_2012-08-03.csv'),
        pd.read_csv('data/chronos_forecast_2012-08-03_2012-10-12.csv')
    ]
    windows = [window1, window2, window3, window4]

    if args.model == 'TiDE':
        find_best_params(dataframe, windows, tests, chronos_forecasts, TiDEModel, get_tide_params,
                         model_name=args.model, approach=args.approach)
    elif args.model == 'TSMixer':
        find_best_params(dataframe, windows, tests, chronos_forecasts, TSMixerModel, get_tsmixer_params,
                         model_name=args.model, approach=args.approach)
    else:
        raise ValueError(f"Model '{args.model}' is not valid.")
