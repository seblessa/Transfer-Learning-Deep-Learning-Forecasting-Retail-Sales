from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import MissingValuesFiller
from sklearn.metrics import root_mean_squared_error
from chronos import ChronosPipeline
import matplotlib.pyplot as plt
from darts import TimeSeries
from typing import Tuple
import pandas as pd
import numpy as np
import torch


def combine_predictions(forecast: pd.DataFrame, residuals_forecast: pd.DataFrame) -> pd.DataFrame:
    # Concatenate the two dataframes
    combined_df = pd.concat([forecast, residuals_forecast])

    # Group by 'unique_id' and TIME_COL and sum the forecast values
    return combined_df.groupby(['unique_id', "Date"]).agg({
        'forecast_lower': 'sum',
        'forecast': 'sum',
        'forecast_upper': 'sum'
    }).reset_index()


def chronos_forecast(
        model: ChronosPipeline, data: pd.DataFrame, horizon: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates forecast with Chronos
    Args:
        model (ChronosPipeline): pre-trained model
        data (pd.DataFrame): historical data
        horizon (int): forecast horizon
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: lower, mid and upper forecast values
    """
    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    context = torch.tensor(data["Weekly_Sales"].tolist())
    forecast = model.predict(
        context, horizon
    )  # shape [num_series, num_samples, prediction_length]

    return np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)


def convert_forecast_to_pandas(
        forecast: list, holdout_set: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert forecast to pandas data frame
    Args:
        forecast (list): list with lower, mid and upper bounds
        holdout_set (pd.DataFrame): data frame with dates in forecast horizon
    Returns:
        pd.DataFrame: forecast in pandas format
    """

    forecast_pd = holdout_set[["unique_id", "Date"]].copy()
    forecast_pd["forecast_lower"] = forecast[0]
    forecast_pd["forecast"] = forecast[1]
    forecast_pd["forecast_upper"] = forecast[2]

    return forecast_pd


def create_darts_list_of_timeseries(
        train: pd.DataFrame
) -> list:
    return TimeSeries.from_group_dataframe(
        df=train,
        group_cols=["Store", "Dept", "Type", "Size"],
        time_col="Date",
        value_cols="Weekly_Sales",
        freq="W-FRI",
        fill_missing_dates=True,
        fillna_value=0)


def create_dynamic_covariates(train_darts: list, dataframe: pd.DataFrame, forecast_horizon: int,
                              w_pred: bool = False) -> list:
    dynamic_covariates = []

    # Ensure the Date column is in datetime format in both dataframes
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    for serie in train_darts:
        # Extract the unique_id from the series to filter the DataFrame
        unique_id = serie.static_covariates['unique_id'].item()

        # Filter the DataFrame for the current series based on unique_id
        filtered_df = dataframe[dataframe['unique_id'] == unique_id]

        # Generate time-related covariates
        covariate = datetime_attribute_timeseries(
            serie,
            attribute="month",
            one_hot=True,
            cyclic=False,
            add_length=forecast_horizon
        ).stack(
            datetime_attribute_timeseries(
                serie,
                attribute="week",
                one_hot=True,
                cyclic=False,
                add_length=forecast_horizon
            )
        )

        if w_pred:
            covariate = covariate.stack(
                TimeSeries.from_dataframe(
                    filtered_df,
                    time_col="Date",
                    value_cols=["IsHoliday", 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
                                'forecast'],
                    freq=serie.freq,
                    fill_missing_dates=True,
                    fillna_value=0
                )
            )
        else:
            covariate = covariate.stack(
                TimeSeries.from_dataframe(
                    filtered_df,
                    time_col="Date",
                    value_cols=["IsHoliday", 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'],
                    freq=serie.freq,
                    fill_missing_dates=True,
                    fillna_value=0
                )
            )

        # Add dynamic covariates that need interpolation
        dyn_cov_interp = TimeSeries.from_dataframe(
            filtered_df,
            time_col="Date",
            value_cols=['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'],
            freq=serie.freq,
            fill_missing_dates=True
        )

        covariate = covariate.stack(MissingValuesFiller().transform(dyn_cov_interp))

        dynamic_covariates.append(covariate)

    return dynamic_covariates


def rmse_evaluation(prediction: pd.DataFrame, actuals: pd.DataFrame) -> float:
    # Ensure input dataframes have the correct datetime format for 'Date'
    prediction['Date'] = pd.to_datetime(prediction['Date'])
    actuals['Date'] = pd.to_datetime(actuals['Date'])

    # Merge predictions with actuals on 'Date' and 'unique_id'
    prediction_w_rmse = pd.merge(prediction, actuals[['Date', 'Weekly_Sales', 'unique_id']],
                                 on=['Date', 'unique_id'], how='left')

    # Calculate squared error # TODO: calculate rmse based on scaled values
    rmse = root_mean_squared_error(prediction_w_rmse['Weekly_Sales'], prediction_w_rmse['forecast'])

    return round(rmse, 2)


def mape_evaluation(prediction: pd.DataFrame, actuals: pd.DataFrame) -> list:
    # Convert 'Date' columns to datetime if they aren't already
    prediction['Date'] = pd.to_datetime(prediction['Date'])
    actuals['Date'] = pd.to_datetime(actuals['Date'])

    # Merging prediction and actual sales data on 'Date' and 'unique_id'
    prediction_w_mape = pd.merge(prediction, actuals[['Date', 'Weekly_Sales', 'unique_id']],
                                 on=['Date', 'unique_id'], how='left')

    # Calculating MAPE
    prediction_w_mape['MAPE'] = abs(prediction_w_mape['forecast'] - prediction_w_mape['Weekly_Sales']) / \
                                prediction_w_mape['Weekly_Sales']

    # Group by 'Date' and calculate the mean MAPE for each group
    weekly_mape = prediction_w_mape.groupby('Date')['MAPE'].mean().tolist()

    # Ensuring the list is rounded to two decimal places
    weekly_mape = [round(x, 2) for x in weekly_mape]

    return weekly_mape


def plot_model_comparison(model_forecasts, actuals, top10_stores=None):
    model_names = ['Chronos', 'TS-Mixer', 'Tide', 'TS-Mixer Chronos',
                   'Tide Chronos', 'TSMixer with Chronos pred in dynamic covariate',
                   'Tide with Chronos pred in dynamic covariate']

    n_weeks = 10  # Assume there are always 10 weeks
    n_models = len(model_forecasts)

    # Prepare data structure to hold MAPE values for each model and week
    weekly_mapes = np.zeros((n_weeks, n_models))

    # Loop through each model's forecasts
    for model_idx, forecasts in enumerate(model_forecasts):
        # Calculate the MAPE for each time window for the current model
        for time_window_idx, model_prediction in enumerate(forecasts):
            # If top10_stores is provided, filter both prediction and actuals DataFrames
            if top10_stores is not None:
                model_prediction = model_prediction[model_prediction['unique_id'].isin(top10_stores['unique_id'])]
                actual_window = actuals[time_window_idx]
                actual_window = actual_window[actual_window['unique_id'].isin(top10_stores['unique_id'])]
            else:
                actual_window = actuals[time_window_idx]

            mape_values = mape_evaluation(model_prediction, actual_window)
            # Accumulate MAPE values to calculate the mean later
            weekly_mapes[:, model_idx] += np.array(mape_values)

        # Calculate mean MAPE across all time windows for the current model
        weekly_mapes[:, model_idx] /= len(forecasts)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    # Create an x-axis marker for each week
    indices = np.arange(n_weeks)
    bar_width = 0.1  # Make bars narrower

    # Set the gap between groups
    group_gap = 0.2  # Large gap between groups
    # Total width of one group of bars
    group_width = n_models * bar_width + (n_models - 1) * 0.02  # Small space between bars in the same group

    # Plot each model's MAPE on the graph
    for i in range(n_models):
        ax.bar(indices + i * bar_width + group_gap * indices, weekly_mapes[:, i], width=bar_width, label=model_names[i])

    ax.set_xlabel('Week')
    ax.set_ylabel('Mean MAPE')
    ax.set_title(
        f"Mean MAPE by Model and Week of the {'top 10 stores' if top10_stores is not None else 'top 500 stores'}")
    ax.set_xticks(indices + group_width / 2 + group_gap * indices)
    ax.set_xticklabels([f'Week {i + 1}' for i in range(n_weeks)])
    ax.legend()

    plt.show()


def plot_multiple_forecasts(
        actuals_data: pd.DataFrame, forecast_data_list: list, title: str, y_label: str, x_label: str,
        forecast_horizon: int, interval: bool = False, top3_stores: pd.DataFrame = None
) -> None:
    """
    Create time series plot of actuals vs multiple forecasts.
    Args:
        actuals_data: DataFrame with actual sales data.
        forecast_data_list: List of tuples (forecast DataFrame, model name).
        title: Title for the chart.
        y_label: Y-axis label.
        x_label: X-axis label.
        forecast_horizon: Number of weeks ahead for the forecast.
        interval: If True, plot prediction interval.
        top3_stores: Optional DataFrame of top stores by unique_id. If provided, filters data to these stores.
    """

    # Filter data for top 10 stores if provided
    if top3_stores is not None:
        actuals_data = actuals_data[actuals_data['unique_id'].isin(top3_stores['unique_id'])]
        forecast_data_list = [(fd[fd['unique_id'].isin(top3_stores['unique_id'])], name)
                              for fd, name in forecast_data_list]

    # Define a list of colors for each model
    colors = ['tomato', 'forestgreen', 'royalblue', 'purple', 'yellow', 'orange', 'pink', 'brown', 'grey', 'cyan']

    # Cut the actuals_data to include only relevant weeks
    actuals_data = actuals_data[
        actuals_data['Date'] >= actuals_data['Date'].max() - pd.DateOffset(weeks=forecast_horizon + 3)]

    plt.figure(figsize=(20, 5))
    plt.plot(
        actuals_data["Date"],
        actuals_data["Weekly_Sales"],
        color="black",
        label="Historical Data",
    )

    for i, (forecast_data, model_name) in enumerate(forecast_data_list):
        plt.plot(
            forecast_data["Date"],
            forecast_data["forecast"],
            color=colors[i],
            label=model_name + " Forecast",
        )

        if interval:
            plt.fill_between(
                forecast_data["Date"],
                forecast_data["forecast_lower"],
                forecast_data["forecast_upper"],
                color=colors[i],
                alpha=0.3,
                label=model_name + " 80% Prediction Interval",
            )

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def transform_predictions_to_pandas(predictions, target: str, pred_list: list, quantiles: list,
                                    convert: bool = True) -> pd.DataFrame:
    """
    Receives as list of predictions and transform it in a data frame
    Args:
        predictions (list): list with predictions
        target (str): column to forecast
        pred_list (list): list with test df to extract time series id
    Returns
        pd.DataFrame: data frame with date, forecast, forecast_lower, forecast_upper and id
        :param convert:
    """

    pred_df_list = []

    for p, pdf in zip(predictions, pred_list):
        temp = (
            p.quantile_df(quantiles[1])
            .reset_index()
            .rename(columns={f"{target}_{quantiles[1]}": "forecast"})
        )
        temp["forecast_lower"] = p.quantile_df(quantiles[0]).reset_index()[f"{target}_{quantiles[0]}"]
        temp["forecast_upper"] = p.quantile_df(quantiles[2]).reset_index()[f"{target}_{quantiles[2]}"]

        # add unique id
        temp["unique_id"] = str(int(list(pdf.static_covariates_values())[0][0])) + '-' + str(
            int(list(pdf.static_covariates_values())[0][1]))

        if convert:
            # convert negative predictions into 0
            temp[["forecast", "forecast_lower", "forecast_upper"]] = temp[
                ["forecast", "forecast_lower", "forecast_upper"]
            ].clip(lower=0)

        pred_df_list.append(temp)

    return pd.concat(pred_df_list)
