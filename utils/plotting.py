import os
import uuid
import matplotlib.pyplot as plt
import pandas as pd


def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.show()


def plot_future_rate(stock_data, preds, n_future_days, currency_pair):
    save_dir = "temp/"
    filename = f"forex_prediction_{uuid.uuid4().hex}.png"
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 5))

    actual_x = stock_data.index[-100:]
    actual_y = stock_data["Close"].values[-100:]

    plt.plot(actual_x, actual_y, label="Actual", linestyle='-', color='blue',
             marker='o', markersize=2)

    last_index = actual_x[-1]

    pred_x = pd.date_range(start=last_index, periods=n_future_days + 1, freq='D')
    pred_y = [actual_y[-1][0]] + list(preds)

    plt.plot(pred_x, pred_y, label="Predicted", marker='o', linestyle='-',
             color='red', markersize=2)

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Forex Rate Prediction: {currency_pair}")

    plt.xticks(rotation=90)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.legend()

    plt.savefig(save_path, dpi=300)
    return save_path


def plot_current_rate(stock_data, currency_pair):
    save_dir = "temp/"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"forex_current_{uuid.uuid4().hex}.png"
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 5))

    x = stock_data.index
    y = stock_data.values

    plt.plot(
        x, y,
        label="Current rate",
        linestyle='-',
        marker='o',
        markersize=2
    )

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Current Forex Rate: {currency_pair}")

    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()

    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path
