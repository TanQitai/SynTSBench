import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ast
from typing import Tuple
from numpy.polynomial.legendre import Legendre
from numpy.polynomial.chebyshev import Chebyshev
from layers.RevIN import RevIN


def generate_legendre_basis(length, n_basis):
    """
    Generates Legendre polynomial basis functions.
    """
    x = np.linspace(-1, 1, length)  # Legendre polynomials are defined on [-1, 1]
    legendre_basis = np.zeros((length, n_basis))
    for i in range(n_basis):
        legendre_poly = Legendre.basis(i)
        legendre_basis[:, i] = legendre_poly(x)
    return legendre_basis


def generate_polynomial_basis(length, n_basis):
    """
    Generates standard polynomial basis functions.
    """
    return np.concatenate(
        [
            np.power(np.arange(length, dtype=float) / length, i)[None, :]
            for i in range(n_basis)
        ]
    ).T


def generate_changepoint_basis(length, n_basis):
    """
    Generates changepoint basis functions with automatically spaced changepoints.
    """
    x = np.linspace(0, 1, length)[:, None]  # Shape: (length, 1)
    changepoint_locations = np.linspace(0, 1, n_basis + 1)[1:][
        None, :
    ]  # Shape: (1, n_basis)
    return np.maximum(0, x - changepoint_locations)


def generate_chebyshev_basis(length, n_basis):
    """
    Generates Chebyshev polynomial basis functions.
    """
    x = np.linspace(-1, 1, length)
    chebyshev_basis = np.zeros((length, n_basis))
    for i in range(n_basis):
        chebyshev_poly = Chebyshev.basis(i)
        chebyshev_basis[:, i] = chebyshev_poly(x)
    return chebyshev_basis


def get_basis(length, n_basis, basis):
    basis_dict = {
        "legendre": generate_legendre_basis,
        "polynomial": generate_polynomial_basis,
        "changepoint": generate_changepoint_basis,
        "chebyshev": generate_chebyshev_basis,
    }
    return basis_dict[basis](length, n_basis + 1)


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, out_features: int = 1):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.out_features = out_features

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, self.backcast_size:self.backcast_size + self.forecast_size]
        return backcast, forecast


class TrendBasis(nn.Module):
    def __init__(
        self,
        n_basis: int,
        backcast_size: int,
        forecast_size: int,
        out_features: int = 1,
        basis="polynomial",
    ):
        super().__init__()
        self.n_basis = n_basis
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.out_features = out_features
        
        # Get basis functions
        backcast_basis = get_basis(backcast_size, n_basis, basis)
        forecast_basis = get_basis(forecast_size, n_basis, basis)
        
        self.register_buffer('backcast_basis', torch.FloatTensor(backcast_basis))
        self.register_buffer('forecast_basis', torch.FloatTensor(forecast_basis))

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # theta: [B, n_basis]
        backcast = torch.matmul(theta, self.backcast_basis.T)  # [B, backcast_size]
        forecast = torch.matmul(theta, self.forecast_basis.T)  # [B, forecast_size]
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(
        self,
        harmonics: int,
        backcast_size: int,
        forecast_size: int,
        out_features: int = 1,
    ):
        super().__init__()
        self.harmonics = harmonics
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.out_features = out_features
        
        # Create harmonic basis
        backcast_basis = []
        forecast_basis = []
        
        for i in range(harmonics):
            # Cosine and sine components
            backcast_cos = np.cos(2 * np.pi * (i + 1) * np.arange(backcast_size) / backcast_size)
            backcast_sin = np.sin(2 * np.pi * (i + 1) * np.arange(backcast_size) / backcast_size)
            backcast_basis.extend([backcast_cos, backcast_sin])
            
            forecast_cos = np.cos(2 * np.pi * (i + 1) * np.arange(forecast_size) / forecast_size)
            forecast_sin = np.sin(2 * np.pi * (i + 1) * np.arange(forecast_size) / forecast_size)
            forecast_basis.extend([forecast_cos, forecast_sin])
        
        self.register_buffer('backcast_basis', torch.FloatTensor(np.column_stack(backcast_basis)))
        self.register_buffer('forecast_basis', torch.FloatTensor(np.column_stack(forecast_basis)))

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # theta: [B, 2*harmonics]
        backcast = torch.matmul(theta, self.backcast_basis.T)  # [B, backcast_size]
        forecast = torch.matmul(theta, self.forecast_basis.T)  # [B, forecast_size]
        return backcast, forecast


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]


class NBEATSBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        dropout_prob: float,
        activation: str,
    ):
        super().__init__()
        
        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        activ = getattr(nn, activation)()
        
        # Build MLP layers
        layers = []
        prev_size = input_size
        
        for hidden_size in mlp_units:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activ)
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size
        
        # Output layer for theta
        layers.append(nn.Linear(prev_size, n_theta))
        
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # insample_y: [B, L]
        theta = self.layers(insample_y)  # [B, n_theta]
        backcast, forecast = self.basis(theta)
        return backcast, forecast


class Model(nn.Module):
    """NBEATS Model adapted for Time-Series-Library framework
    
    The Neural Basis Expansion Analysis for Time Series (NBEATS), is a simple and yet
    effective architecture, it is built with a deep stack of MLPs with the doubly
    residual connections.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Model parameters
        self.n_harmonics = getattr(configs, 'n_harmonics', 10)
        self.n_basis = getattr(configs, 'n_basis', 4)
        self.basis = getattr(configs, 'basis', 'polynomial')
        
        # Parse string parameters
        stack_types_str = getattr(configs, 'stack_types', '["seasonality", "trend", "identity"]')
        if isinstance(stack_types_str, str):
            self.stack_types = ast.literal_eval(stack_types_str)
        else:
            self.stack_types = stack_types_str
            
        n_blocks_str = getattr(configs, 'n_blocks', '[3, 3, 3]')
        if isinstance(n_blocks_str, str):
            self.n_blocks = ast.literal_eval(n_blocks_str)
        else:
            self.n_blocks = n_blocks_str
            
        mlp_units_str = getattr(configs, 'mlp_units', '[[512, 512], [512, 512], [512, 512]]')
        if isinstance(mlp_units_str, str):
            self.mlp_units = ast.literal_eval(mlp_units_str)
        else:
            self.mlp_units = mlp_units_str
            
        self.dropout_prob_theta = getattr(configs, 'dropout_prob_theta', 0.0)
        self.activation = getattr(configs, 'activation', 'ReLU')
        
        shared_weights_str = getattr(configs, 'shared_weights', 'False')
        if isinstance(shared_weights_str, str):
            self.shared_weights = shared_weights_str.lower() == 'true'
        else:
            self.shared_weights = shared_weights_str
        
        # Use RevIN for normalization
        self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=False)
        
        # Create stacks
        self.stacks = nn.ModuleList()
        
        for i, (stack_type, n_block, mlp_unit) in enumerate(
            zip(self.stack_types, self.n_blocks, self.mlp_units)
        ):
            # Create basis for this stack
            if stack_type == 'seasonality':
                basis = SeasonalityBasis(
                    harmonics=self.n_harmonics,
                    backcast_size=self.seq_len,
                    forecast_size=self.pred_len,
                    out_features=1,
                )
                n_theta = 2 * self.n_harmonics
            elif stack_type == 'trend':
                basis = TrendBasis(
                    n_basis=self.n_basis,
                    backcast_size=self.seq_len,
                    forecast_size=self.pred_len,
                    out_features=1,
                    basis=self.basis,
                )
                n_theta = self.n_basis + 1
            elif stack_type == 'identity':
                basis = IdentityBasis(
                    backcast_size=self.seq_len,
                    forecast_size=self.pred_len,
                    out_features=1,
                )
                n_theta = self.seq_len + self.pred_len
            else:
                raise ValueError(f"Stack type {stack_type} not supported")
            
            # Create blocks for this stack
            stack_blocks = nn.ModuleList()
            for j in range(n_block):
                if self.shared_weights and j > 0:
                    # Use the same block as the first one
                    block = stack_blocks[0]
                else:
                    block = NBEATSBlock(
                        input_size=self.seq_len,
                        n_theta=n_theta,
                        mlp_units=mlp_unit,
                        basis=basis if not self.shared_weights else 
                               (TrendBasis(self.n_basis, self.seq_len, self.pred_len, 1, self.basis) 
                                if stack_type == 'trend' else
                                SeasonalityBasis(self.n_harmonics, self.seq_len, self.pred_len, 1)
                                if stack_type == 'seasonality' else
                                IdentityBasis(self.seq_len, self.pred_len, 1)),
                        dropout_prob=self.dropout_prob_theta,
                        activation=self.activation,
                    )
                stack_blocks.append(block)
            self.stacks.append(stack_blocks)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [B, L, D]
        # Apply RevIN normalization
        x = self.revin_layer(x, 'norm')
        
        batch_size, seq_len, n_vars = x.shape
        
        # Initialize forecasts
        forecast = torch.zeros(batch_size, self.pred_len, n_vars, device=x.device)
        
        # Process each variable independently
        for d in range(n_vars):
            residual = x[:, :, d].clone()  # [B, L]
            var_forecast = torch.zeros(batch_size, self.pred_len, device=x.device)
            
            # Apply each stack
            for stack_blocks in self.stacks:
                for block in stack_blocks:
                    backcast, forecast_block = block(residual)  # [B, L], [B, H]
                    residual = residual - backcast
                    var_forecast = var_forecast + forecast_block
            
            forecast[:, :, d] = var_forecast
        
        # Apply RevIN denormalization
        forecast = self.revin_layer(forecast, 'denorm')
        
        return forecast
