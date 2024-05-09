from darts.dataprocessing.transformers import StaticCovariatesTransformer, MissingValuesFiller
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.pipeline import Pipeline
from chronos import ChronosPipeline
from darts.models import TiDEModel
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from darts import TimeSeries
from typing import Tuple
import seaborn as sns
import pandas as pd
import numpy as np
import torch


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


def chronos_prediction(
        window: tuple,  # (start_date, end_date)
        dataframe: pd.DataFrame,  # historical data
        architecture: tuple,  # (tiny or large, cpu or gpu)
        forecast_horizon: int  # time period to forecast ahead after the end of the window
) -> pd.DataFrame:
    train = dataframe[(dataframe["Date"] <= window[0])]
    test = dataframe[(dataframe["Date"] > window[0]) & (dataframe["Date"] <= window[1])]

    train_darts = create_darts_list_of_timeseries(train)

    # Load the Chronos pipeline
    pipeline = ChronosPipeline.from_pretrained(
        architecture[0],
        device_map=architecture[1],
        torch_dtype=torch.bfloat16)

    forecast = []
    for ts in train_darts:
        # Forecast
        lower, mid, upper = chronos_forecast(pipeline, ts.pd_dataframe().reset_index(), forecast_horizon)
        forecast.append(convert_forecast_to_pandas([lower, mid, upper], test[
            test['unique_id'] == str(int(list(ts.static_covariates_values())[0][0])) + '-' + str(
                int(list(ts.static_covariates_values())[0][1]))]))
    # Convert list to data frames
    prediction = pd.concat(forecast)
    prediction.to_csv('data/chronos_forecast_' + str(window[0].date().strftime('%Y-%m-%d')) + '|' + str(
        window[1].date().strftime('%Y-%m-%d')) + '.csv', index=False)

    return prediction


def tide_prediction(
        window: tuple,  # (start_date, end_date)
        dataframe: pd.DataFrame,  # historical data
        forecast_horizon: int  # time period to forecast ahead after the end of the window
) -> pd.DataFrame:

    train = dataframe[(dataframe["Date"] <= window[0])]
    test = dataframe[(dataframe["Date"] > window[0]) & (dataframe["Date"] <= window[1])]

    train_darts = create_darts_list_of_timeseries(train)

    dynamic_covariates = []
    for serie in train_darts:
        # add the month and week as a covariate
        covariate = datetime_attribute_timeseries(
            serie,
            attribute="month",
            one_hot=True,
            cyclic=False,
            add_length=forecast_horizon,
        )
        covariate = covariate.stack(
            datetime_attribute_timeseries(
                serie,
                attribute="week",
                one_hot=True,
                cyclic=False,
                add_length=forecast_horizon,
            )
        )

        store = serie.static_covariates['Store'].item()
        dept = serie.static_covariates['Dept'].item()

        # create covariates to fill with 0
        covariate = covariate.stack(
            TimeSeries.from_dataframe(dataframe[(dataframe['Store'] == store) & (dataframe['Dept'] == dept)], time_col="Date",
                                      value_cols=["IsHoliday", 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'],
                                      freq="W-FRI", fill_missing_dates=True, fillna_value=0)
        )

        # create covariates to fill with interpolation
        dyn_cov_interp = TimeSeries.from_dataframe(dataframe[(dataframe['Store'] == store) & (dataframe['Dept'] == dept)],
                                                   time_col="Date",
                                                   value_cols=['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'],
                                                   freq="W-FRI",
                                                   fill_missing_dates=True)

        covariate = covariate.stack(MissingValuesFiller().transform(dyn_cov_interp))

        dynamic_covariates.append(covariate)

    SCALER = Scaler()
    TRANSFORMER = StaticCovariatesTransformer()
    PIPELINE = Pipeline([SCALER, TRANSFORMER])

    # scale covariates
    dynamic_covariates_transformed = SCALER.fit_transform(dynamic_covariates)

    # scale data and transform static covariates
    data_transformed = PIPELINE.fit_transform(train_darts)

    TiDE_params = {
        "input_chunk_length": len(train["Date"].unique()) - forecast_horizon,  # number of weeks to lookback
        "output_chunk_length": forecast_horizon,  # number of weeks to forecast
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "decoder_output_dim": 1,
        "hidden_size": 15,
        "temporal_width_past": 4,
        "temporal_width_future": 4,
        "temporal_decoder_hidden": 26,
        "dropout": 0.1,
        "batch_size": 16,
        "n_epochs": 50,
        "likelihood": QuantileRegression(quantiles=[0.25, 0.5, 0.75]),
        "random_state": 42,
        "use_static_covariates": True,
        "optimizer_kwargs": {"lr": 1e-3},
        "use_reversible_instance_norm": False,
    }

    model = TiDEModel(**TiDE_params)
    model.fit(data_transformed, future_covariates=dynamic_covariates_transformed, verbose=False)
    pred = PIPELINE.inverse_transform(
        model.predict(n=forecast_horizon, series=data_transformed, future_covariates=dynamic_covariates_transformed,
                      num_samples=50))
    tide_forecast = transform_predictions_to_pandas(pred, "Weekly_Sales", train_darts, [0.25, 0.5, 0.75])
    tide_forecast.to_csv('data/tide_forecast.csv' + str(window[0].date().strftime('%Y-%m-%d')) + '|' + str(
        window[1].date().strftime('%Y-%m-%d')) + '.csv', index=False)
    return tide_forecast





def evaluation_metrics(prediction: pd.Series, actuals: pd.Series) -> float:
    prediction_w_mape = pd.merge(prediction, actuals.loc[:, ["Date", "Weekly_Sales", "unique_id"]],
                                 on=["Date", "unique_id"], how="left")
    prediction_w_mape["MAPE"] = abs(prediction_w_mape["forecast"] - prediction_w_mape["Weekly_Sales"]) / \
                                prediction_w_mape["Weekly_Sales"]
    return round(prediction_w_mape["MAPE"].mean(), 2)


def plot_model_comparison(dataframe: pd.DataFrame) -> None:
    """
    Bar plot comparison between models
    Args:
        dataframe (pd.DataFrame): data with actuals and forecats for both models
    """

    tide_model = dataframe.rename(
        columns={"TiDE": "forecast"}
    )
    tide_model["model"] = "TiDE"
    tide_model["MAPE"] = (
            abs(tide_model["Weekly_Sales"] - tide_model["forecast"]) / tide_model["Weekly_Sales"]
    )

    chronos_tiny_model = dataframe.rename(
        columns={"Chronos Tiny": "forecast"}
    )
    chronos_tiny_model["model"] = "Chronos Tiny"
    chronos_tiny_model["MAPE"] = (
            abs(chronos_tiny_model["Weekly_Sales"] - chronos_tiny_model["forecast"])
            / chronos_tiny_model["Weekly_Sales"]
    )

    chronos_large_model = dataframe.rename(
        columns={"Chronos Large": "forecast"}
    )
    chronos_large_model["model"] = "Chronos Large"
    chronos_large_model["MAPE"] = (
            abs(chronos_large_model["Weekly_Sales"] - chronos_large_model["forecast"])
            / chronos_large_model["Weekly_Sales"]
    )

    plt.rcParams["figure.figsize"] = (20, 5)
    ax = sns.barplot(
        data=pd.concat(
            [tide_model, chronos_tiny_model, chronos_large_model]
        ),
        x="Date",
        y="MAPE",
        hue="model",
        palette=["#dd4fe4", "#070620", "#fa7302"],
    )
    plt.title("Comparison between TiDE and Chronos in Walmart data")
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()


def plot_multiple_forecasts(
        actuals_data: pd.DataFrame, forecast_data_list: list, title: str, y_label: str, x_label: str,
        forecast_horizon: int, interval: bool = False
) -> None:
    """
    Create time series plot of actuals vs multiple forecasts
    Args:
        :param forecast_horizon:
        :param interval: if True, plot prediction interval
        :param actuals_data: actual data from the dataset
        :param forecast_data_list: list of tuples (forecast dataframe, model name)
        :param title: title for chart
        :param x_label
        :param y_label
    """

    # Define a list of colors for each model
    colors = ['tomato', 'forestgreen', 'royalblue']

    # Cut the actuals_data to 5 weeks before the prediction date
    actuals_data = actuals_data[
        actuals_data['Date'] >= actuals_data['Date'].max() - pd.DateOffset(weeks=forecast_horizon + 3)]

    plt.figure(figsize=(20, 5))
    plt.plot(
        actuals_data["Date"],
        actuals_data["Weekly_Sales"],
        color="black",
        label="historical data",
    )

    for i, (forecast_data, model_name) in enumerate(forecast_data_list):
        plt.plot(
            forecast_data["Date"],
            forecast_data["forecast"],
            color=colors[i],
            label=model_name + " forecast",
        )

        if interval:
            plt.fill_between(
                forecast_data["Date"],
                forecast_data["forecast_lower"],
                forecast_data["forecast_upper"],
                color=colors[i],
                alpha=0.3,
                label=model_name + " 80% prediction interval",
            )

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_actuals_forecast(
        actuals_data: pd.DataFrame, forecast_data: pd.DataFrame, title: str
) -> None:
    """
    Create time series plot actuals vs forecast
    Args:
        actuals_data (pd.DataFrame): actual data
        forecast_data (pd.DataFrame): forecast
        title (str): title for chart
    """

    plt.figure(figsize=(20, 5))
    plt.plot(
        actuals_data["Date"],
        actuals_data["Weekly_Sales"],
        color="royalblue",
        label="historical data",
    )
    plt.plot(
        forecast_data["Date"],
        forecast_data["forecast"],
        color="tomato",
        label="median forecast",
    )
    plt.fill_between(
        forecast_data["Date"],
        forecast_data["forecast_lower"],
        forecast_data["forecast_upper"],
        color="tomato",
        alpha=0.3,
        label="80% prediction interval",
    )
    plt.legend()
    plt.grid()
    plt.title(title)
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
