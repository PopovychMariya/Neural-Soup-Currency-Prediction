import time
import numpy as np
from curl_cffi import requests
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import pickle
from constants import CURRENCY_PAIR_TO_TICKER


def get_data(ticker='EURUAH=X'):
    session = requests.Session(impersonate="chrome")
    session.verify = False

    stock_data = yf.download(
                            tickers=ticker,
                            interval="1d",
                            start="2020-11-25",
                            end="2025-11-25",
                            progress=False,
                            session=session,
                        )
    return stock_data


def prepare_data(df, window_size=60):
    scaler = StandardScaler()
    df = df[["Close"]]
    scaled = scaler.fit_transform(df.values)
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


def save_scaler(file, scaler):
    with open(file, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(file):
    with open(file, "rb") as f:
        loaded_scaler = pickle.load(f)

    return loaded_scaler


def prepare_currency_pair_data(ticker):
    df = get_data(ticker)
    df_filename = ticker + "_stock_data.csv"
    scaler_filename = ticker + "_scaler.pkl"
    df.to_csv(df_filename, index=True, encoding="utf-8")
    X, y, scaler = prepare_data(df)

    save_scaler(scaler_filename, scaler)

    return X, y


if __name__ == "__main__":
    for ticker in CURRENCY_PAIR_TO_TICKER.values():
        X, y = prepare_currency_pair_data(ticker)
        np.save(f"{ticker}_X.npy", X)
        np.save(f"{ticker}_y.npy", y)

        time.sleep(3)
