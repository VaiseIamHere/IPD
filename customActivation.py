import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    """Gausian Error Linear Units (GELU) Activation function."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return F.gelu(data)

class InputScaledGELU(nn.Module):
    """GELU with input scaling."""

    def __init__(self, num_channels) -> None:
        super().__init__()
        alpha_init = torch.randn(num_channels)
        alpha_init = alpha_init / alpha_init.norm(p=2)  # L2 norm

        self.alpha = nn.Parameter(alpha_init)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.alpha.view(1, -1, 1, 1) * data)

class OutputScaledGELU(nn.Module):
    """GELU with Output scaling."""
    def __init__(self, num_channels) -> None:
        super().__init__()
        alpha_init = torch.randn(num_channels)
        alpha_init = alpha_init / alpha_init.norm(p=2)  # L2 norm

        self.alpha = nn.Parameter(alpha_init)

    def forward(self, data: torch.Tensor) -> torch.Tensor:    
        return self.alpha.view(1, -1, 1, 1) * F.gelu(data)

class MultiScaledGELU(nn.Module):
    """GELU with Input as well as Output with scaling, with added bias."""
    def __init__(self, num_channels) -> None:
        super().__init__()
        alpha_init = torch.randn(num_channels)
        alpha_init = alpha_init / alpha_init.norm(p=2)  # L2 norm

        beta_init = torch.randn(num_channels)
        beta_init = beta_init / beta_init.norm(p=2)  # L2 norm

        gamma_init = torch.randn(num_channels)
        gamma_init = gamma_init / gamma_init.norm(p=2)  # L2 norm

        self.alpha = nn.Parameter(alpha_init)
        self.beta = nn.Parameter(beta_init)
        self.gamma = nn.Parameter(gamma_init)

    def forward(self, data: torch.Tensor) -> torch.Tensor:    
        inputData = self.alpha.view(1, -1, 1, 1) * data + self.beta.view(1, -1, 1, 1)
        outputActivation = self.gamma.view(1, -1, 1, 1) * F.gelu(inputData)
        return outputActivation

class PolynomialGELU(nn.Module):
    """GELU with parameterized appoximate parametric function."""
    def __init__(self, num_channels) -> None:
        super().__init__()
        alpha_init = torch.randn(num_channels)
        alpha_init = alpha_init / alpha_init.norm(p=2)  # L2 norm

        beta_init = torch.randn(num_channels)
        beta_init = beta_init / beta_init.norm(p=2)  # L2 norm

        gamma_init = torch.randn(num_channels)
        gamma_init = gamma_init / gamma_init.norm(p=2)  # L2 norm

        delta_init = torch.randn(num_channels)
        delta_init = delta_init / delta_init.norm(p=2)  # L2 norm

        self.alpha = nn.Parameter(alpha_init)
        self.beta = nn.Parameter(beta_init)
        self.gamma = nn.Parameter(gamma_init)
        self.delta = nn.Parameter(delta_init)

        self.k = (2 ** 0.5)/torch.pi

        self.tanh = nn.Tanh()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data_cubed = torch.pow(data, 3)
        polynomial = self.alpha.view(1, -1, 1, 1) * 0.044715 * data_cubed + self.beta.view(1, -1, 1, 1) * data + self.gamma.view(1, -1, 1, 1)
        product = self.k * polynomial
        return self.delta.view(1, -1, 1, 1) * 0.5 * self.tanh(product)
