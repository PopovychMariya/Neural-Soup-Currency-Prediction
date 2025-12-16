from core import get_model_prediction
from datetime import datetime, timedelta


if __name__ == "__main__":
    currency_pair = "EUR-UAH"

    end_date = datetime.now().date() - timedelta(days=0)
    start_date = datetime.now().date() - timedelta(days=120)
    n_days = 100
    pred, plot_path = get_model_prediction(
        currency_pair,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        n_days)
