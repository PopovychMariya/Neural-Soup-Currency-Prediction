from .constants import CURRENCY_PAIR_TO_TICKER
import numpy as np


def get_full_data():
    X_list = []
    y_list = []
    for ticker in CURRENCY_PAIR_TO_TICKER.values():
        x = np.load(f"data/{ticker}_X.npy")
        y = np.load(f"data/{ticker}_y.npy")
        X_list.append(x)
        y_list.append(y)

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
