import torch
import torch.nn as nn
import numpy as np


def update_gaussian(mu1, mu2, logsig1, logsig2):
    _eps = 1e-5
    sig1, sig2 = torch.exp(logsig1), torch.exp(logsig2)
    mu = (mu1*sig2 + mu2*sig1) / (sig1 + sig2)
    sig = sig1 * sig2 / (sig1 + sig2 + _eps)
    logsig = torch.log(sig + _eps)
    return mu, logsig


class PedSpace(nn.Module):
    def __init__(self,
                 modalities=[],
                 m_settings=[],
                 n_proto=20,
                 proto_dim=128,
                 ) -> None:
        super().__init__()
        self.modalities = modalities
        self.n_proto = n_proto
        self.proto_dim = proto_dim
        for m in self.modalities:
            