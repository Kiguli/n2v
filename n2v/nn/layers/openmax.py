"""OpenMax wrapper: open-set recognition variant of softmax."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenMax(nn.Module):
    """OpenMax activation for open-set classification.

    Implements the Weibull-recalibrated activation of Bendale & Boult
    (2016). For reachability purposes the layer behaves as an affine
    recalibration on the activation vector followed by softmax over an
    augmented ``num_classes + 1`` dimension.
    """

    def __init__(self, num_classes: int, alpha: int = 10):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = int(alpha)
        self.mean_vec = nn.Parameter(torch.zeros(num_classes, num_classes), requires_grad=False)
        self.weibull_scale = nn.Parameter(torch.ones(num_classes), requires_grad=False)
        self.weibull_shape = nn.Parameter(torch.ones(num_classes), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank = torch.argsort(x, dim=-1, descending=True)
        recalib = torch.ones_like(x)
        for i in range(min(self.alpha, x.size(-1))):
            w = float(self.alpha - i) / float(self.alpha)
            idx = rank[..., i:i + 1]
            recalib.scatter_(-1, idx, w)
        recalib_x = x * recalib
        unknown = (x - recalib_x).sum(dim=-1, keepdim=True)
        augmented = torch.cat([recalib_x, unknown], dim=-1)
        return F.softmax(augmented, dim=-1)
