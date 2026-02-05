import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.userModel.user_model import UserModel


class SimpleMLP(UserModel):
    def __init__(self, input_dim, hidden_dim=64, **kwargs):
        super(SimpleMLP, self).__init__(feature_columns=[], y_columns=[], **kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out.squeeze(-1)

    def get_loss(self, x, y, score=None):
        y_pred = self.forward(x)

        # Mean Squared Error or BCE
        loss = F.mse_loss(y_pred, y)

        return loss
