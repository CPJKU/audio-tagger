import torch
import torch.nn as nn
import numpy as np


class SmallNN(nn.Module):

    def __init__(self, in_shape, n_classes, n_hidden=64, device='cpu'):
        super(SmallNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_shape, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),

            nn.Linear(n_hidden, n_classes),
            nn.Sigmoid()
        )

        self.device = device

        self.mean = 0
        self.std = 1

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):

        cfa, cft, llf = x

        if len(cfa.shape) != 2:
            cfa = np.expand_dims(cfa, 0)

        x = np.hstack([cfa, cft, llf])

        # normalize x
        x = (x - self.mean) / self.std

        x = torch.from_numpy(x).float().to(self.device)

        with torch.no_grad():
            probs = self.net(x)

        return probs.cpu().numpy()

    def predict(self, x):
        probs = self.predict_proba(x)
        probs[probs < 0.5] = 0
        probs[probs >= 0.5] = 1

        return np.asarray(probs, dtype=np.int32)

    def set_stats(self, mean, std):

        self.mean = mean
        self.std = std


def load_masp_nn_model(model_path, stats_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SmallNN(39, 3, device=device)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage))
    model.to(device)
    model.eval()
    model.set_stats(**np.load(stats_path))

    return model