# ZAAI
Curricular Internship at ZAAI

# Summary

The aim of the project is to use prediction models in the context of time series forecasting. 
The dataset used is the [Walmart Sales Forecasting dataset](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data).

The project goal is to compared the models to see which one performs better.
The models used are:

- Chronos
- TiDE
- TSMixer

Additionally, we plan to explore a hybrid approach by combining these models.
Initially, we'll generate predictions using Chronos and then employ its residuals as training input for the TSMixer and TiDE models. Then use these residuals predictions to refine the original Chronos forecast.
Furthermore, we're going to try to use Chronos predictions as covariates in the TSMixer and TiDE models to get refined forecasts.

In the end we'll compare the models using the following Mean Absolute Percentage Error (MAPE) metric. 

# Versões

The version of the operating system used to develop this project is:
- macOS Sonoma 14.5

Python Versions:
- 3.12


# Requirements

To keep everything organized and simple,
we will use [MiniConda](https://docs.conda.io/projects/miniconda/en/latest/) to manage our environments.

To create an environment with the required packages for this project, run the following commands:

```bash
conda create -n venv python
```

Then we need to install the requirements:

```bash
pip install -r requirements.txt
```

if there is a bug with lightbm do this:
- `brew install libomp`

# Results

You can see the notebook here: [notebook.ipynb](notebook.ipynb).


