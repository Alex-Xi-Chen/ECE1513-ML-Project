"""
Data loader: download daily CAD exchange-rate data from the Bank of Canada
Valet API and cache it locally as CSV files.
"""

import os
import pandas as pd
import requests

import config


def fetch_series(series_name: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a single series from the Bank of Canada Valet API.

    Returns a DataFrame with columns ['date', series_name].
    """
    url = f"{config.BOC_API_BASE}/{series_name}/json"
    params = {
        "start_date": start,
        "end_date": end,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    observations = data["observations"]

    records = []
    for obs in observations:
        date = obs["d"]
        value = obs[series_name]["v"]
        records.append({"date": date, "rate": float(value)})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"rate": series_name})
    return df


def download_all(
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download exchange-rate data for all configured currency pairs.

    Returns a dict mapping currency code (e.g. 'USD') to its DataFrame.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    datasets: dict[str, pd.DataFrame] = {}

    for currency, series_name in config.CURRENCY_PAIRS.items():
        csv_path = os.path.join(config.DATA_DIR, f"{currency}_CAD.csv")

        if os.path.exists(csv_path):
            print(f"[data_loader] Loading cached {currency}/CAD from {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["date"])
        else:
            print(f"[data_loader] Downloading {currency}/CAD ({series_name}) ...")
            df = fetch_series(series_name, start, end)
            if save:
                df.to_csv(csv_path, index=False)
                print(f"[data_loader] Saved to {csv_path}")

        datasets[currency] = df

    return datasets


def load_csv(currency: str) -> pd.DataFrame:
    """Load a previously-saved CSV for a given currency pair."""
    csv_path = os.path.join(config.DATA_DIR, f"{currency}_CAD.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run download_all() first."
        )
    return pd.read_csv(csv_path, parse_dates=["date"])


if __name__ == "__main__":
    datasets = download_all()
    for cur, df in datasets.items():
        print(f"{cur}/CAD: {len(df)} rows, range {df['date'].min()} – {df['date'].max()}")
