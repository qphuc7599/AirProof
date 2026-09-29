"""Causal, independently implemented ESN--TNN research comparator.

Architecture: Bonas and Castruccio (2025), JRSS C, doi:10.1093/jrsssc/qlaf007.
The authors' repository is Env-an-Stat-group/25.Bonas.JRSSC. This is not a claim
of reproducing their numerical results: lag/horizon, fitting splits, ensemble
size and optimizer budget are explicit experiment parameters. In particular,
PCA is fitted ONLY on the reservoir-training prefix, never on future states.
PyTorch is optional and is imported only when the transformer is requested.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy import linalg, sparse


@dataclass(frozen=True)
class DESNConfig:
    units: int = 500
    layers: int = 3
    reduced_units: int = 10
    spectral_radii: tuple[float, ...] = (0.4, 1.0, 1.0)
    leak: float = 0.49
    sparsity: float = 0.1
    ridge: float = 0.1
    lag: int = 1
    ensemble: int = 20
    washout: int = 12

    def validate(self) -> None:
        if self.units < 2 or self.layers < 1 or self.ensemble < 1:
            raise ValueError("positive layer/ensemble counts and at least two units required")
        if self.layers != len(self.spectral_radii) or min(self.spectral_radii) <= 0:
            raise ValueError("one positive spectral radius per layer required")
        if not 1 <= self.reduced_units <= self.units:
            raise ValueError("reduced dimension must lie in [1, units]")
        if not 0 < self.leak <= 1 or not 0 < self.sparsity <= 1 or self.ridge <= 0:
            raise ValueError("invalid reservoir regularization")
        if self.lag < 1 or self.washout < 0:
            raise ValueError("lag must be positive and washout nonnegative")


@dataclass
class _Layer:
    recurrent: sparse.csr_matrix
    input_weights: np.ndarray
    mean: np.ndarray | None = None
    components: np.ndarray | None = None
    score_scale: np.ndarray | None = None


@dataclass
class _Member:
    layers: list[_Layer]
    readout: np.ndarray


def _checked_data(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[1] < 1 or not np.isfinite(data).all():
        raise ValueError("expected finite time-by-station observations")
    return data


class DeepEchoStateEnsemble:
    """Three-level leaky sparse DESN, EOF/PCA reduction and ridge readout.

    ``predict_series`` returns a one-step forecast for target index t based on
    observations through t-1. Parameters must have been fitted before that target;
    values inside the fit prefix are fitted predictions, not held-out forecasts.
    All preprocessing statistics and principal components stay frozen after fit.
    """

    def __init__(self, config: DESNConfig = DESNConfig(), *, seed: int = 0):
        config.validate()
        self.config, self.seed = config, int(seed)
        self.members: list[_Member] = []

    def _inputs(self, values: np.ndarray) -> np.ndarray:
        # Row j predicts t = lag+1+j. The last row predicts one beyond values.
        z = (values - self.mean) / self.scale
        lag = self.config.lag
        return np.column_stack((np.ones(len(z) - lag), z[:-lag], z[lag:]))

    def _new_layer(self, inputs: int, level: int, rng: np.random.Generator) -> _Layer:
        cfg = self.config
        # Degenerate zero-radius draws are possible for very small test fixtures.
        for _ in range(20):
            recurrent = rng.normal(size=(cfg.units, cfg.units))
            recurrent *= rng.random(recurrent.shape) < cfg.sparsity
            radius = float(np.max(np.abs(linalg.eigvals(recurrent))))
            if radius > 1e-10:
                break
        else:
            raise ValueError("degenerate reservoir draw: increase units/sparsity")
        recurrent *= cfg.spectral_radii[level] / radius
        weights = rng.normal(size=(cfg.units, inputs))
        weights *= rng.random(weights.shape) < cfg.sparsity
        return _Layer(sparse.csr_matrix(recurrent), weights)

    def _features(self, inputs: np.ndarray, layers: list[_Layer], *, fit: bool) -> np.ndarray:
        reduced = []
        for level, layer in enumerate(layers):
            injection = inputs @ layer.input_weights.T
            state = np.zeros(self.config.units)
            history = np.empty((len(inputs), self.config.units))
            for t in range(len(inputs)):
                activated = np.tanh(layer.recurrent @ state + injection[t])
                state = (1 - self.config.leak) * state + self.config.leak * activated
                history[t] = state
            if level == len(layers) - 1:
                return np.column_stack((history, *reduced)) if reduced else history
            if fit:
                layer.mean = history.mean(axis=0)
                centered = history - layer.mean
                _, _, vt = linalg.svd(centered, full_matrices=False, check_finite=False)
                layer.components = vt[: self.config.reduced_units].T
                scores = centered @ layer.components
                layer.score_scale = np.maximum(scores.std(axis=0, ddof=1), 1e-8)
            else:
                scores = (history - layer.mean) @ layer.components
            scores = scores / layer.score_scale
            reduced.append(np.tanh(scores))
            inputs = np.column_stack((np.ones(len(scores)), scores))
        raise AssertionError("no reservoir layers")

    def fit(
        self, training: np.ndarray, *, progress: Callable[[int, int], None] | None = None
    ) -> DeepEchoStateEnsemble:
        data = _checked_data(training)
        cfg = self.config
        if len(data) - cfg.lag - 1 <= max(cfg.reduced_units, cfg.washout + 1):
            raise ValueError("training prefix too short for lag, PCA or washout")
        self.fit_length, self.stations = data.shape
        self.mean = float(data.mean())
        self.scale = max(float(data.std(ddof=1)), 1e-8)
        inputs = self._inputs(data)[:-1]  # no target beyond the fitting prefix
        targets = (data[cfg.lag + 1 :] - self.mean) / self.scale
        self.members.clear()
        for member_seed in np.random.SeedSequence(self.seed).spawn(cfg.ensemble):
            rng = np.random.default_rng(member_seed)
            layers = [self._new_layer(inputs.shape[1] if level == 0 else cfg.reduced_units + 1,
                                      level, rng) for level in range(cfg.layers)]
            features = self._features(inputs, layers, fit=True)[cfg.washout :]
            gram = features.T @ features + cfg.ridge * np.eye(features.shape[1])
            readout = linalg.solve(gram, features.T @ targets[cfg.washout :],
                                   assume_a="pos", check_finite=False)
            self.members.append(_Member(layers, readout))
            if progress:
                progress(len(self.members), cfg.ensemble)
        return self

    def predict_series(self, observations: np.ndarray, *, include_next: bool = False) -> np.ndarray:
        data = _checked_data(observations)
        if not self.members or data.shape[1] != self.stations:
            raise ValueError("fit the model first and retain station ordering")
        if len(data) <= self.config.lag:
            raise ValueError("insufficient lag history")
        inputs = self._inputs(data)
        prediction = np.zeros((len(inputs), self.stations))
        for member in self.members:
            prediction += self._features(inputs, member.layers, fit=False) @ member.readout
        prediction = self.mean + self.scale * prediction / len(self.members)
        # The target domain is nonnegative concentration, identically for all methods.
        prediction = np.maximum(prediction, 0)
        output = np.full((len(data) + int(include_next), self.stations), np.nan)
        output[self.config.lag + 1 :] = prediction if include_next else prediction[:-1]
        return output

    def metadata(self) -> dict:
        return {"architecture": "DESN-PCA-ridge", "config": asdict(self.config),
                "seed": self.seed, "fit_length": self.fit_length, "stations": self.stations,
                "scaling": "global training-prefix mean and sample SD",
                "PCA_scope": "reservoir-training prefix only",
                "target_horizon": 1, "latest_input_for_target_t": "t-1"}


def make_transformer(*, channels: int = 4, dropout: float = 0.1):
    """One encoder/decoder, two heads, 4d feed-forward, learned positions, skips.

    Encoder features are [intercept, public observed history]; decoder features
    are [intercept, DESN forecast]. Never place a target value in decoder input.
    """
    import torch
    from torch import nn

    if channels < 2 or channels % 2 or not 0 <= dropout < 1:
        raise ValueError("an even channel count and dropout in [0,1) are required")

    class EchoTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_position = nn.Embedding(1024, channels)
            self.decoder_position = nn.Embedding(1024, channels)
            self.encoder_input = nn.Linear(2, channels)
            self.decoder_input = nn.Linear(2, channels)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(channels, 2, 4 * channels, dropout,
                                           batch_first=True), 1, enable_nested_tensor=False)
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(channels, 2, 4 * channels, dropout,
                                           batch_first=True), 1)
            self.output = nn.Linear(channels, 1)

        def forward(self, source, forecast):
            if source.ndim != 3 or forecast.ndim != 3 or source.shape[-1] != 2 or forecast.shape[-1] != 2:
                raise ValueError("expected batch-by-time-by-two feature tensors")
            if max(source.shape[1], forecast.shape[1]) > 1024:
                raise ValueError("sequence exceeds learned positional capacity")
            src = self.encoder_input(source)
            memory = self.encoder(src + self.encoder_position(
                torch.arange(source.shape[1], device=source.device))) + src
            dec = self.decoder_input(forecast)
            positions = self.decoder_position(torch.arange(forecast.shape[1], device=forecast.device))
            mask = torch.triu(torch.ones(forecast.shape[1], forecast.shape[1],
                                         dtype=torch.bool, device=forecast.device), diagonal=1)
            decoded = self.decoder(dec + positions, memory, tgt_mask=mask) + dec
            return self.output(decoded)

    return EchoTransformer()


def station_windows(
    observations: np.ndarray, predictions: np.ndarray, indices: np.ndarray, *,
    history: int, mean: float, scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build target-t samples using observed t-history:t and a forecast of t.

    Flattening order is (target time, station). No target/training labels are read
    by this function: modifying observations[t:] cannot change sample t inputs.
    """
    values = _checked_data(observations)
    forecasts = np.asarray(predictions, dtype=float)
    times = np.asarray(indices, dtype=int)
    if forecasts.shape != values.shape or history < 1 or scale <= 0 or not len(times):
        raise ValueError("invalid forecast windows")
    if times.min() < history or times.max() >= len(values):
        raise ValueError("window crosses observed history boundary")
    src_values = np.stack([values[t - history : t].T for t in times])
    src_values = ((src_values - mean) / scale).reshape(-1, history)
    pred_values = ((forecasts[times] - mean) / scale).reshape(-1, 1)
    if not np.isfinite(pred_values).all():
        raise ValueError("forecast is unavailable for a requested target")
    source = np.stack((np.ones_like(src_values), src_values), axis=-1).astype(np.float32)
    decoder = np.stack((np.ones_like(pred_values), pred_values), axis=-1).astype(np.float32)
    return source, decoder


def convex_weight(esn: np.ndarray, transformer: np.ndarray, truth: np.ndarray) -> float:
    """Validation-only MSE optimum w for (1-w)*ESN + w*TNN, 0<=w<=1."""
    a, b, y = (np.asarray(x, dtype=float) for x in (esn, transformer, truth))
    if a.shape != b.shape or a.shape != y.shape or not all(np.isfinite(x).all() for x in (a, b, y)):
        raise ValueError("finite matched validation arrays required")
    difference = b - a
    denominator = float(np.sum(difference**2))
    return float(np.clip(np.sum((y - a) * difference) / denominator, 0, 1)) if denominator else 0.0
