# ECE1513 ML Project – Predicting Canadian Dollar Exchange Rates

Predicting Canadian Dollar (CAD) exchange rates against major global currencies (USD, EUR, CNY) using machine learning. Daily exchange-rate data is sourced from the **Bank of Canada Valet API**.

## Project Structure

```
ECE1513-ML-Project/
├── main.py                  # Run the full pipeline (download → train → evaluate)
├── config.py                # All hyperparameters & paths in one place
├── requirements.txt         # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Download & cache data from Bank of Canada API
│   ├── preprocessing.py     # Feature engineering & train/val/test splitting
│   ├── models.py            # Model definitions (Linear Regression, SVR, MLP)
│   ├── train.py             # Training loops (sklearn + PyTorch)
│   ├── evaluate.py          # Metrics (MAE, RMSE, R²) and plotting
│   └── utils.py             # Seed setting, directory helpers
├── data/                    # Auto-created; cached CSV files
├── results/
│   ├── figures/             # Auto-created; prediction & residual plots
│   └── results_summary.csv  # Auto-created; metrics table
└── reference doc/
    ├── main.tex             # Project proposal
    └── Report_Template.tex  # Report template (NeurIPS format)
```

## Problem Description

Exchange-rate forecasting is formulated as a **supervised regression** task:

- **Input**: engineered features from historical daily rates (lagged observations, rolling means/stds, percentage changes, cyclical date encodings).
- **Output**: exchange rate *h* days ahead (default *h* = 1).

## Models

| Model | Description |
|---|---|
| **Linear Regression** | Baseline – ordinary least squares via scikit-learn |
| **SVR** | Support Vector Regression with RBF kernel |
| **MLP** | Multi-Layer Perceptron (PyTorch) with BatchNorm, Dropout, and early stopping |

## Evaluation Metrics

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R²** – Coefficient of Determination

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This will:
1. Download daily exchange-rate data from the Bank of Canada (cached to `data/`).
2. Engineer lag, rolling, and calendar features.
3. Split chronologically into train (70%) / validation (15%) / test (15%).
4. Train Linear Regression, SVR, and MLP on each currency pair.
5. Print metrics and save plots to `results/figures/` and a summary CSV.

## Configuration

All key settings live in `config.py`:

- **Currency pairs** – add or remove pairs in `CURRENCY_PAIRS`.
- **Feature engineering** – adjust `LAG_DAYS`, `ROLLING_WINDOWS`, `FORECAST_HORIZON`.
- **SVR hyperparameters** – `SVR_PARAMS`.
- **MLP hyperparameters** – `MLP_PARAMS` (hidden sizes, learning rate, epochs, early stopping patience, etc.).

## Data Source

[Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs) – free, public, no API key required.

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for package versions.
