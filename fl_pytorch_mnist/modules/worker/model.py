import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Build the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.to(device)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

    def set_weights(self, weights):
        for layer, w in weights.items():
            self.state_dict()[layer].copy_(torch.FloatTensor(w))

    def get_weights(self):
        weights = {}
        for layer, w in self.state_dict().items():
            # weights[layer] = w.detach().cpu().tolist()
            weights[layer] = np.array(w.detach().cpu().numpy())
        return weights
