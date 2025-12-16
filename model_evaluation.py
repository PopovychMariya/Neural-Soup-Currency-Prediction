import argparse
import warnings
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.exceptions import InconsistentVersionWarning

from constants import CURRENCY_PAIR_TO_TICKER
from core import get_yf_data
from data.load_yf import load_scaler
from model import init_model


DEFAULT_HORIZONS: Tuple[int, ...] = (1, 3, 7, 10, 14, 21, 30)
DEFAULT_WINDOW_SIZE = 60
DEFAULT_LOOKBACK_DAYS = 365

# Silence noisy library warnings (pickle version mismatch, yfinance future changes, etc.).
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quick metrics on the saved model using historical rates."
    )
    parser.add_argument(
        "--currency-pairs",
        nargs="+",
        default=list(CURRENCY_PAIR_TO_TICKER.keys()),
        help="Currency pairs to evaluate (default: all pairs in constants.py).",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Forecast horizons (in days) to score.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Sliding window size used for predictions.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD for evaluation window.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD for evaluation window (default: today).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="If no start-date is provided, fetch this many days back from end-date.",
    )

    return parser.parse_args()


def to_dates(start_date: str, end_date: str, lookback_days: int) -> Tuple[str, str]:
    end = (
        datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_date
        else datetime.now().date()
    )
    start = (
        datetime.strptime(start_date, "%Y-%m-%d").date()
        if start_date
        else end - timedelta(days=lookback_days)
    )

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def multi_step_forecast(
    model: torch.nn.Module, window: np.ndarray, horizon: int, device: torch.device
) -> np.ndarray:
    """
    Generate autoregressive predictions for the requested horizon.

    Args:
        model: Trained LSTM model.
        window: 1D numpy array of scaled values with length == window_size.
        horizon: Number of days to predict ahead.
        device: Torch device to run inference on.
    """
    history = window.copy()
    preds = []

    for _ in range(horizon):
        input_seq = torch.tensor(
            history.reshape(1, -1, 1), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            pred_scaled = model(input_seq).item()
        preds.append(pred_scaled)
        history = np.append(history[1:], pred_scaled)

    return np.array(preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100

    return {
        "mae": mae,
        "mse": mse,
        "mape": mape,
        "rmse": rmse,
    }


def evaluate_currency_pair(
    model: torch.nn.Module,
    currency_pair: str,
    start_date: str,
    end_date: str,
    horizons: Iterable[int],
    window_size: int,
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    ticker = CURRENCY_PAIR_TO_TICKER[currency_pair]
    scaler = load_scaler(f"data/{ticker}_scaler.pkl")

    stock_data = get_yf_data(start_date, end_date, ticker)
    close_series = stock_data["Close"].dropna()

    if len(close_series) < window_size + max(horizons):
        raise ValueError(
            f"Not enough data for {currency_pair}. "
            f"Need at least {window_size + max(horizons)} points, got {len(close_series)}."
        )

    scaled_close = scaler.transform(close_series.values.reshape(-1, 1)).flatten()
    reports: Dict[int, Dict[str, float]] = {}

    for horizon in horizons:
        preds_scaled: List[float] = []
        targets_scaled: List[float] = []

        for idx in range(window_size, len(scaled_close) - horizon + 1):
            window = scaled_close[idx - window_size : idx]
            forecast = multi_step_forecast(model, window, horizon, device)
            preds_scaled.append(forecast[-1])
            targets_scaled.append(scaled_close[idx + horizon - 1])

        preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
        targets_scaled_arr = np.array(targets_scaled).reshape(-1, 1)

        preds = scaler.inverse_transform(preds_scaled_arr).flatten()
        targets = scaler.inverse_transform(targets_scaled_arr).flatten()

        metrics = compute_metrics(targets, preds)
        metrics["samples"] = len(targets)
        reports[horizon] = metrics

    return reports


def load_model(device: torch.device) -> torch.nn.Module:
    model = init_model(from_checkpoint=True)
    model.to(device)
    model.eval()
    return model


def print_report(
    currency_pair: str,
    report: Dict[int, Dict[str, float]],
    start_date: str,
    end_date: str,
) -> None:
    print(f"\n{currency_pair} | window: {start_date} -> {end_date}")
    for horizon in sorted(report.keys()):
        metrics = report[horizon]
        print(
            f"  + Horizon {horizon:>3}d "
            f"[{metrics['samples']:>4} samples]: "
            f"MAE={metrics['mae']:.6f} "
            f"MSE={metrics['mse']:.6f} "
            f"MAPE={metrics['mape']:.2f}%"
        )


def main() -> None:
    args = parse_args()
    start_date, end_date = to_dates(args.start_date, args.end_date, args.lookback_days)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    for pair in args.currency_pairs:
        try:
            report = evaluate_currency_pair(
                model=model,
                currency_pair=pair,
                start_date=start_date,
                end_date=end_date,
                horizons=args.horizons,
                window_size=args.window_size,
                device=device,
            )
            print_report(pair, report, start_date, end_date)
        except Exception as exc:
            print(f"\n{pair}: evaluation failed -> {exc}")


if __name__ == "__main__":
    main()
