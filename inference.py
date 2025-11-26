from core import get_model_prediction
from datetime import datetime, timedelta


if __name__ == "__main__":
    currency_pair = 'UAH-EUR'

    end_date = datetime.now().date()
    start_date = datetime.now().date() - timedelta(days=120)
    n_days = 10
    pred = get_model_prediction(
        currency_pair,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        n_days)
