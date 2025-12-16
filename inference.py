from core import get_model_prediction, get_history_rate
from datetime import datetime, timedelta


if __name__ == "__main__":
    currency_pair = "USD-UAH"

    end_date = datetime.now().date() - timedelta(days=0)
    start_date = datetime.now().date() - timedelta(days=120)
    n_days = 100
    pred, plot_path = get_model_prediction(
        currency_pair,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        n_days)

    history_plot_path = get_history_rate(
        currency_pair,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
