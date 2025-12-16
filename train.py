import torch
from torch import nn
from torch.optim import Adam
from utils.data import get_full_data

from .constants import CHECKPOINT_PATH
from .utils.plotting import plot_losses
from dataset import ForexDataset
from model import RatePredictor
from torch.utils.data import DataLoader, random_split


batch_size = 32
epochs = 500
learning_rate = 1e-4
weight_decay = 1e-3
test_ratio = 0.2

data, labels = get_full_data()
dataset = ForexDataset(data, labels)

test_size = int(len(dataset) * test_ratio)
train_size = len(dataset) - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = RatePredictor()
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


best_test_loss = float('inf')
train_losses = []
test_losses = []

for e in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        batch_y = batch_y.view_as(output)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss /= train_size
    train_losses.append(epoch_loss)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            output = model(batch_X)
            batch_y = batch_y.view_as(output)
            loss = criterion(output, batch_y)
            test_loss += loss.item() * batch_X.size(0)
    test_loss /= test_size
    test_losses.append(test_loss)

    print(f"Epoch {e+1}/{epochs} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f}")

    if test_loss < best_test_loss * 0.95:
        best_test_loss = test_loss
        torch.save({
            'epoch': e + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'test_loss': test_loss
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved at epoch {e+1} with test loss {test_loss:.6f}")

plot_losses(train_losses, test_losses)



