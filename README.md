# Chronos

**Steps to run the code:**

1. Create a virtual environment with python
`conda create --name myenv python=3.10.13`
3. Activate in your new virtual environment
`conda activate myenv`
4. Install the required requirements
    - `pip install git+https://github.com/amazon-science/chronos-forecasting.git`
    - `pip install -r requirements.txt`
    - 
if there is a bug with lightbm do this:
- `brew install libomp`
And then repeat step 4
5. Create a folder called `/data` under `Chronos/` and add your data (https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data?select=train.csv)
6. Run the notebook


# ZAAI
Curricular Intership at ZAAI


# Summary

The aim of the project is
to compare the Chronos and TiDE models in the context of time series forecasting. 
The dataset used is the Walmart Sales Forecasting dataset.


RELATÓRIO:

O que é transfer learning
Explicar forecasting
Definir retail sales

Explicar detalhadamente o dataset utilizado

Explicar os modelos usados (TiDE, Chronos)

Explicar as possibilidades de utilização:

- Simple Chronos (just sales),
- Simple TiDE (all the covariates),
- Chronos with the residuals added from the trained TiDE with the dataset created by me based on the original dataset, with the residuals from Chronos.

Concluir a melhor possibilidade de combinação






TUNING NOTES:




Parameters found: {'input_chunk_length': 22, 'output_chunk_length': 10, 'num_encoder_layers': 1, 'num_decoder_layers': 1, 'decoder_output_dim': 1, 'hidden_size': 10, 'temporal_width_past': 2, 'temporal_width_future': 2, 'temporal_decoder_hidden': 20, 'dropout': 0.001, 'batch_size': 8, 'n_epochs': 10, 'likelihood': QuantileRegression(quantiles: Optional[List[float]] = None), 'random_state': 42, 'use_static_covariates': True, 'optimizer_kwargs': {'lr': 1e-05}, 'use_reversible_instance_norm': False}
MAPE found: 0.18


Parameters found: {'input_chunk_length': 22, 'output_chunk_length': 10, 'num_encoder_layers': 1, 'num_decoder_layers': 1, 'decoder_output_dim': 1, 'hidden_size': 10, 'temporal_width_past': 2, 'temporal_width_future': 2, 'temporal_decoder_hidden': 20, 'dropout': 0.001, 'batch_size': 8, 'n_epochs': 15, 'likelihood': QuantileRegression(quantiles: Optional[List[float]] = None), 'random_state': 42, 'use_static_covariates': True, 'optimizer_kwargs': {'lr': 1e-05}, 'use_reversible_instance_norm': False}
MAPE found: 0.17















# Versões

The versions of the operating systems used to develop and test this application are:
- Fedora 38
- macOS Sonoma 14.0

Python Versions:
- 3.10.0


# Requirements

To keep everything organized and simple,
we will use [MiniConda](https://docs.conda.io/projects/miniconda/en/latest/) to manage our environments.

To create an environment with the required packages for this project, run the following commands:

```bash
conda create -n GymEnv python=3.10 pytorch::pytorch torchvision torchaudio -c pytorch
```

Then we need to install the requirements:

```bash
pip install -r requirements.txt
```

# Results

##### The experiments are in this [notebook](notebook.ipynb).




|       TQC in the original environment:        |       TQC in the wrapped environment        |
|:---------------------------------------------:|:-------------------------------------------:|
|  ![TQC_original.gif](media/TQC_original.gif)  | ![TQC_wrapped.gif](media%2FTQC_wrapped.gif) |



|       TRPO in the wrapped environment:        |       PPO in the wrapped environment:       |
|:---------------------------------------------:|:-------------------------------------------:|
| ![TRPO_wrapped.gif](media%2FTRPO_wrapped.gif) | ![PPO_wrapped.gif](media%2FPPO_wrapped.gif) |
