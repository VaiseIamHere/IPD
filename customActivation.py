import torch
import torch.nn as nn

class GELU(nn.Module):
    """Gausian Error Linear Units (GELU) Activation function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        cdf = 0.5 * (1 + torch.erf(data / 2.0**0.5))
        return data * cdf
