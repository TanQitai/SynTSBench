"""Common signal generation and augmentation utilities used by SynTSBench.

This module consolidates helper functions that were previously duplicated
across multiple dataset generation notebooks. All notebooks under
`SynTSBench/dataset` can now simply import what they need from here.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess

__all__ = [
    # Signal generators
    "generate_linear_signal",
    "generate_quadratic_signal",
    "generate_exponential_signal",
    "generate_logarithmic_signal",
    "generate_logistic_signal",
    "generate_gompertz_signal",
    "generate_power_law_signal",
    "generate_step_signal",
    "generate_piecewise_linear_signal",
    "generate_gaussian_signal",
    "generate_sine_wave",
    "generate_composite_sine",
    "generate_cosine_wave",
    "generate_triangle_wave",
    "generate_sawtooth_wave",
    "generate_square_wave",
    "generate_sinc_wave",
    "generate_exp_sine_wave",
    "generate_autoregressive_signal",
    "generate_arma_signal",
    "generate_random_walk",
    "generate_white_noise",
    "generate_garch_signal",
    # Noise/anomaly helpers
    "add_noise_by_snr",
    "add_poisson_noise",
    "add_point_anomalies",
    "add_pulse_anomalies",
    # IO + viz
    "save_signal",
    "plot_signals",
]


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------

def generate_linear_signal(n: int, slope: float = 10, intercept: float = 0) -> np.ndarray:
    """Generate a linear signal with given slope and intercept."""
    x = np.arange(n)
    return slope * x + intercept


def generate_quadratic_signal(n: int, a: float = 1, b: float = 1, c: float = 1) -> np.ndarray:
    """Generate a quadratic signal: ``a * x**2 + b * x + c``."""
    x = np.arange(n)
    return a * x**2 + b * x + c


def generate_exponential_signal(n: int, rate: float = 0.01, a: float = 1) -> np.ndarray:
    """Generate an exponential signal: ``a * e^(rate * x) + 1``."""
    x = np.arange(n)
    return a * np.exp(rate * x) + 1


def generate_logarithmic_signal(n: int, base: float = 2) -> np.ndarray:
    """Generate a logarithmic signal with specified base."""
    x = np.arange(1, n + 1)
    return np.log(x) / np.log(base)


def generate_logistic_signal(n: int, L: float = 1, k: float = 0.1, t0: float | None = None) -> np.ndarray:
    """Generate a logistic (sigmoid) signal representing S-shaped growth."""
    if t0 is None:
        t0 = n / 2

    x = np.arange(n)
    return L / (1 + np.exp(-k * (x - t0)))


def generate_gompertz_signal(n: int, a: float = 1, b: float = 2, k: float = 0.1) -> np.ndarray:
    """Generate a Gompertz function signal."""
    x = np.arange(n)
    return a * np.exp(-b * np.exp(-k * x))


def generate_power_law_signal(n: int, a: float = 1, b: float = 0.5) -> np.ndarray:
    """Generate a power law signal for scaling relationships."""
    x = np.arange(1, n + 1)  # Avoid x=0 if b < 0
    return a * x**b


def generate_step_signal(n: int, t0: int | None = None, c: float = 0, d: float = 1) -> np.ndarray:
    """Generate a step function signal representing a sudden change."""
    if t0 is None:
        t0 = n // 2

    signal_values = np.ones(n) * c
    signal_values[t0:] = d
    return signal_values


def generate_piecewise_linear_signal(
    n: int,
    breakpoints: Sequence[int],
    values: Sequence[float],
) -> np.ndarray:
    """Generate a piecewise linear signal with specified breakpoint values."""
    if len(breakpoints) != len(values):
        raise ValueError("Number of breakpoints must equal number of values")

    bp_list = list(breakpoints)
    val_list = list(values)

    if 0 not in bp_list:
        bp_list = [0] + bp_list
        val_list = [val_list[0]] + val_list
    if n - 1 not in bp_list:
        bp_list = bp_list + [n - 1]
        val_list = val_list + [val_list[-1]]

    bp_values = sorted(zip(bp_list, val_list))
    sorted_bp = [bp for bp, _ in bp_values]
    sorted_values = [val for _, val in bp_values]

    signal_values = np.zeros(n)
    for i in range(len(sorted_bp) - 1):
        start_idx = sorted_bp[i]
        end_idx = sorted_bp[i + 1]
        start_val = sorted_values[i]
        end_val = sorted_values[i + 1]

        if end_idx > start_idx:
            for j in range(start_idx, end_idx + 1):
                t = (j - start_idx) / (end_idx - start_idx)
                signal_values[j] = (1 - t) * start_val + t * end_val

    return signal_values


def generate_gaussian_signal(n: int, a: float = 1, t0: float | None = None, sigma: float | None = None) -> np.ndarray:
    """Generate a Gaussian (bell curve) signal."""
    if t0 is None:
        t0 = n / 2
    if sigma is None:
        sigma = n / 10

    x = np.arange(n)
    return a * np.exp(-((x - t0) ** 2) / (2 * sigma**2))


def generate_sine_wave(n: int, amplitude: float = 1, frequency: float = 0.5, phase: float = 0) -> np.ndarray:
    """Generate a sine wave with specified parameters."""
    x = np.arange(n)
    return amplitude * np.sin(2 * np.pi * frequency * x + phase)


def generate_composite_sine(
    n: int,
    components: Iterable[Tuple[float, float, float] | Tuple[float, float, float, str]],
) -> np.ndarray:
    """Generate a signal composed of multiple sine or cosine waves."""
    x = np.arange(n)
    signal_values = np.zeros(n)

    for component in components:
        if len(component) == 3:
            amplitude, frequency, phase_shift = component
            wave_type = "sin"
        elif len(component) == 4:
            amplitude, frequency, phase_shift, wave_type = component
        else:
            raise ValueError(
                "Each component must be (amplitude, frequency, phase_shift[, wave_type])"
            )

        if wave_type.lower() == "sin":
            signal_values += amplitude * np.sin(2 * np.pi * frequency * x + phase_shift)
        elif wave_type.lower() == "cos":
            signal_values += amplitude * np.cos(2 * np.pi * frequency * x + phase_shift)
        else:
            raise ValueError("wave_type must be 'sin' or 'cos'")

    return signal_values


def generate_cosine_wave(n: int, amplitude: float = 1, frequency: float = 0.5, phase: float = 0) -> np.ndarray:
    """Generate a cosine wave with specified parameters."""
    x = np.arange(n)
    return amplitude * np.cos(2 * np.pi * frequency * x + phase)


def generate_triangle_wave(n: int, amplitude: float = 1, frequency: float = 0.5) -> np.ndarray:
    """Generate a triangle wave with specified parameters."""
    return amplitude * signal.sawtooth(2 * np.pi * frequency * np.arange(n), 0.5)


def generate_sawtooth_wave(
    n: int,
    amplitude: float = 1,
    frequency: float = 0.1,
    phase_shift: float = 0.99,
) -> np.ndarray:
    """Generate a sawtooth wave with consistent peak values."""
    return amplitude * signal.sawtooth(2 * np.pi * frequency * np.arange(n) + phase_shift * np.pi)


def generate_square_wave(n: int, amplitude: float = 1, frequency: float = 0.5) -> np.ndarray:
    """Generate a square wave with specified parameters."""
    return amplitude * signal.square(2 * np.pi * frequency * np.arange(n))


def generate_sinc_wave(n: int, amplitude: float = 1, frequency: float = 1.0) -> np.ndarray:
    """Generate a sinc wave (sin(x)/x)."""
    x = np.linspace(-8 * np.pi / frequency, 8 * np.pi / frequency, n)
    return amplitude * np.sinc(x / np.pi)


def generate_exp_sine_wave(n: int, amplitude: float = 1, frequency: float = 0.5) -> np.ndarray:
    """Generate an exponential sine wave (e^sin(x))."""
    x = np.arange(n)
    return amplitude * np.exp(np.sin(2 * np.pi * frequency * x))


def generate_autoregressive_signal(
    n: int,
    coeffs: Sequence[float] = (0.6, 0.2, 0.15),
    noise_std: float = 1,
) -> np.ndarray:
    """Generate an autoregressive (AR) signal."""
    ar_signal = np.zeros(n)
    order = len(coeffs)
    for i in range(order, n):
        ar_signal[i] = sum(coeff * ar_signal[i - j - 1] for j, coeff in enumerate(coeffs))
        ar_signal[i] += np.random.normal(0, noise_std)
    return ar_signal


def generate_arma_signal(n: int, ar_params: Sequence[float], ma_params: Sequence[float]) -> np.ndarray:
    """Generate an ARMA (AutoRegressive Moving Average) signal."""
    arma_process = ArmaProcess(ar=np.r_[1, -np.array(ar_params)], ma=np.r_[1, np.array(ma_params)])
    return arma_process.generate_sample(nsample=n)


def generate_random_walk(n: int, start: float = 0) -> np.ndarray:
    """Generate a random walk starting from the given value."""
    steps = np.random.normal(0, 1, n)
    return np.cumsum(steps) + start


def generate_white_noise(n: int, mean: float = 0, std: float = 1) -> np.ndarray:
    """Generate white noise with specified mean and standard deviation."""
    return np.random.normal(mean, std, n)


def generate_garch_signal(
    n: int = 1000,
    omega: float = 0.1,
    alpha: float = 0.2,
    beta: float = 0.7,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate GARCH(1,1) time series data."""
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be less than 1 to ensure stationarity")
    if omega <= 0:
        raise ValueError("omega must be greater than 0")

    np.random.seed(seed)

    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        returns[t] = np.random.normal(0, np.sqrt(sigma2[t - 1]))
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    return returns, sigma2


# ---------------------------------------------------------------------------
# Noise/anomaly helpers
# ---------------------------------------------------------------------------

def add_noise_by_snr(signal_values: np.ndarray, snr_db: float = 20) -> np.ndarray:
    """Add Gaussian noise based on signal-to-noise ratio (SNR in dB)."""
    signal_power = np.var(signal_values)
    snr_linear = 10 ** (snr_db / 10)
    noise_variance = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_variance), len(signal_values))
    return signal_values + noise


def add_poisson_noise(signal_values: np.ndarray) -> np.ndarray:
    """Add Poisson noise (signal must be non-negative)."""
    offset = np.min(signal_values)
    if offset < 0:
        signal_shifted = signal_values - offset
    else:
        signal_shifted = signal_values

    noisy_signal = np.random.poisson(np.maximum(signal_shifted, 0))

    if offset < 0:
        noisy_signal = noisy_signal + offset
    return noisy_signal


def _adjust_amplitude_range(signal_range: float, base_range: Tuple[float, float], high_mult: float, low_mult: float) -> Tuple[float, float]:
    if signal_range > 100:
        return base_range[0] * low_mult, base_range[1] * low_mult
    if signal_range < 4:
        return base_range[0] * high_mult, base_range[1] * high_mult
    return base_range


def add_point_anomalies(
    signal_values: np.ndarray,
    anomaly_ratio: float = 0.05,
    base_amplitude_range: Tuple[float, float] = (0.1, 0.3),
    debug: bool = False,
) -> np.ndarray:
    """Add point anomalies with amplitude adaptively adjusted to signal range."""
    signal_range = np.max(signal_values) - np.min(signal_values)
    signal_length = len(signal_values)
    train_length = int(signal_length * 0.8)
    num_anomalies = max(1, int(train_length * anomaly_ratio))

    adjusted = _adjust_amplitude_range(signal_range, base_amplitude_range, high_mult=2, low_mult=0.2)

    if debug:
        print(f"Signal range: {signal_range}")
        print(
            "Adjusted anomaly amplitude range:"
            f" {adjusted[0] * signal_range} to {adjusted[1] * signal_range}"
        )

    anomaly_positions = np.random.choice(train_length, num_anomalies, replace=False)
    anomalous_signal = signal_values.copy()
    for pos in anomaly_positions:
        sign = np.random.choice([-1, 1])
        amplitude = np.random.uniform(adjusted[0], adjusted[1]) * signal_range
        anomalous_signal[pos] += sign * amplitude

    return anomalous_signal


def add_pulse_anomalies(
    signal_values: np.ndarray,
    num_pulses: int = 3,
    pulse_width_range: Tuple[int, int] = (10, 30),
    base_amplitude_range: Tuple[float, float] = (0.1, 0.3),
    debug: bool = False,
) -> np.ndarray:
    """Add pulse anomalies with amplitude adaptively adjusted to signal range."""
    signal_range = np.max(signal_values) - np.min(signal_values)
    signal_length = len(signal_values)
    train_length = int(signal_length * 0.8)

    adjusted = _adjust_amplitude_range(signal_range, base_amplitude_range, high_mult=2, low_mult=0.8)

    if debug:
        print(f"Signal range: {signal_range}")
        print(
            "Adjusted pulse amplitude range:"
            f" {adjusted[0] * signal_range} to {adjusted[1] * signal_range}"
        )

    anomalous_signal = signal_values.copy()

    min_distance = max(pulse_width_range[1], train_length // (num_pulses * 2))
    possible_positions = list(range(0, train_length - pulse_width_range[1]))
    pulse_positions: List[int] = []

    for _ in range(num_pulses):
        while possible_positions:
            pos = np.random.choice(possible_positions)
            if not any(abs(pos - p) < min_distance for p in pulse_positions):
                pulse_positions.append(pos)
                possible_positions = [p for p in possible_positions if abs(p - pos) >= min_distance]
                break
            possible_positions = [p for p in possible_positions if p != pos]
        if not possible_positions:
            break

    for pos in pulse_positions:
        width = np.random.randint(pulse_width_range[0], pulse_width_range[1])
        sign = np.random.choice([-1, 1])
        amplitude = np.random.uniform(adjusted[0], adjusted[1]) * signal_range
        anomalous_signal[pos : pos + width] += sign * amplitude

    return anomalous_signal


# ---------------------------------------------------------------------------
# Persistence and visualization helpers
# ---------------------------------------------------------------------------

def save_signal(
    signal_values: np.ndarray,
    folder: str,
    signal_name: str,
    dataset_id: int | None = None,
    category: str | None = None,
    length: int | None = None,
) -> None:
    """Save a 1-D signal to CSV within the specified folder."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = signal_name
    if dataset_id is not None:
        filename = f"dataset{dataset_id}_{filename}"
    if category is not None:
        filename = f"{filename}_{category}"
    if length is not None:
        filename = f"{filename}_length{length}"

    timestamps = np.arange(len(signal_values))
    df = pd.DataFrame({"date": timestamps, "Feature1": signal_values})
    df.to_csv(os.path.join(folder, f"{filename}.csv"), index=False)


def plot_signals(signal_data_dict: dict[str, np.ndarray]) -> None:
    """Visualize multiple signals in a grid layout."""
    plt.figure(figsize=(15, 20))
    total = len(signal_data_dict)
    cols = 3
    rows = total // cols + int(total % cols != 0)
    for i, (name, signal_values) in enumerate(signal_data_dict.items()):
        plt.subplot(rows, cols, i + 1)
        plt.plot(signal_values)
        plt.title(name)
        plt.grid(True)
    plt.tight_layout()
    plt.show()


