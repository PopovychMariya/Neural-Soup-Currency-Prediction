import torch
from torch import nn
from torch.functional import F
from constants import CHECKPOINT_PATH


class RatePredictor(nn.Module):
    def __init__(self, input_features=1):
        super().__init__()

        self.lstm1 = nn.LSTM(
            hidden_size=100,
            input_size=input_features,
            batch_first=True,
            num_layers=3,
            dropout=0.2)

        self.fc1 = nn.Linear(in_features=100, out_features=25)
        self.fc2 = nn.Linear(in_features=25, out_features=1)
        self.norm = nn.LayerNorm(100)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        output, (h_n, c_n) = self.lstm1(x)
        # x = output[:, -1, :]
        x = self.norm(output[:, -1, :])

        x = F.relu(self.fc1(x))

        x = self.dropout(x)
        x = self.fc2(x)

        return x


def init_model(from_checkpoint=True):
    model = RatePredictor()
    if from_checkpoint:
        model = RatePredictor()
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])

    return model
