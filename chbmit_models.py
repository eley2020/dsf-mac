#!/usr/bin/env python3
"""Neural network models for CHB-MIT seizure prediction experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from train_chbmit_baseline import CNNGRU


CHBMIT_COMMON_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
]


BIOT_18_CHANNEL_ORDER = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "C3-A2",
    "C4-A1",
]


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class AdaptiveSpectralBlock1d(nn.Module):
    """TSLANet-style adaptive spectral filter for multichannel EEG.

    The input and output shapes are ``[batch, time, channels]``. The block
    learns channel-wise complex spectral weights and an adaptive high-energy
    mask, then returns the filtered signal in the time domain. A residual
    connection is applied by the calling model.
    """

    def __init__(
        self,
        channels: int,
        adaptive_filter: bool = True,
        weight_scale: float = 0.02,
    ):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight = nn.Parameter(
            weight_scale * torch.randn(channels, 2, dtype=torch.float32)
        )
        self.complex_weight_high = nn.Parameter(
            weight_scale * torch.randn(channels, 2, dtype=torch.float32)
        )
        self.threshold_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def create_adaptive_high_freq_mask(self, x_fft: torch.Tensor) -> torch.Tensor:
        energy = x_fft.abs().pow(2)
        batch = energy.shape[0]
        median_energy = energy.reshape(batch, -1).median(dim=1, keepdim=True).values
        median_energy = median_energy.reshape(batch, 1, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        hard_mask = (normalized_energy > self.threshold_param).to(dtype=x_fft.real.dtype)
        return (hard_mask - self.threshold_param).detach() + self.threshold_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        time_steps = x.shape[1]
        x_fft = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        filtered_fft = x_fft * weight
        if self.adaptive_filter:
            mask = self.create_adaptive_high_freq_mask(x_fft)
            high_weight = torch.view_as_complex(self.complex_weight_high)
            filtered_fft = filtered_fft + x_fft * mask * high_weight
        filtered = torch.fft.irfft(filtered_fft, n=time_steps, dim=1, norm="ortho")
        return filtered.to(dtype=x.dtype)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            SqueezeExcite1d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class DualStreamFreqNet(nn.Module):
    """Lightweight dual-stream temporal/frequency CNN for seizure prediction.

    Input shape follows the existing dataset convention: ``[batch, time, channels]``.
    This study-specific risk estimator keeps the CNN-GRU baseline interface and
    adds an FFT-magnitude branch inspired by the frequency-aware dual-stream
    principle explored in the authors' prior DUFGNet work. It is not a DUFGNet
    reimplementation: it does not use the 1D/2D U-Nets, DCT channel attention,
    graph integration, or self-supervised pretraining of DUFGNet.
    """

    def __init__(self, channels: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(hidden, hidden, kernel_size=3, stride=2),
        )
        self.frequency = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=3, stride=2),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x_time = x.permute(0, 2, 1)
        temporal_features = self.temporal(x_time)

        freq = torch.fft.rfft(x_time, dim=-1).abs()
        freq = torch.log1p(freq)
        freq_features = self.frequency(freq)

        target_len = min(temporal_features.shape[-1], freq_features.shape[-1])
        temporal_features = temporal_features[..., :target_len]
        freq_features = freq_features[..., :target_len]
        fused = self.fusion(torch.cat([temporal_features, freq_features], dim=1))
        sequence = fused.permute(2, 0, 1)
        _, hidden = self.gru(sequence)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class AttnSleepNet(nn.Module):
    """Compact AttnSleep-inspired EEG risk network.

    The original AttnSleep sleep-staging architecture combines temporal
    convolution, channel/spatial filtering and attention over the downsampled
    sequence.  This adaptation keeps the existing CHB-MIT interface
    (``[batch, time, channels] -> 2 logits``), while using a causal-safe
    sequence encoder and attentive pooling for a single window.  The model is
    deliberately compact so it can be compared with DSF under the same LOOCV
    protocol on the available Titan V GPUs.
    """

    def __init__(self, channels: int, embed_dim: int = 128, heads: int = 4, dropout: float = 0.25):
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError("embed_dim must be divisible by heads")
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=25, stride=4, padding=12, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, stride=2, padding=4, groups=64, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            DepthwiseSeparableBlock(embed_dim, embed_dim, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(embed_dim, embed_dim, kernel_size=5, stride=1),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.pool_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        z = self.temporal_stem(x.permute(0, 2, 1))
        z = self.spatial(z)
        z = self.refine(z).transpose(1, 2)
        z = self.norm(z)
        attended, _ = self.attention(z, z, z, need_weights=False)
        z = self.attn_norm(z + attended)
        weights = torch.softmax(self.pool_score(z).squeeze(-1), dim=1)
        return torch.sum(z * weights.unsqueeze(-1), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class TriBandDualStreamFreqNet(nn.Module):
    """DSF variant with explicit low/normal/high EEG spectral bands.

    The temporal stream is unchanged from :class:`DualStreamFreqNet`.  The
    frequency stream computes an FFT magnitude and routes it through three
    fixed, non-overlapping bands (below ``low_cut_hz``, ``low_cut_hz`` to
    ``high_cut_hz``, and above ``high_cut_hz``).  A small sample-adaptive gate
    combines the three encoded bands before temporal/frequency fusion.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 96,
        dropout: float = 0.25,
        sample_rate: float = 256.0,
        low_cut_hz: float = 0.5,
        high_cut_hz: float = 30.0,
    ):
        super().__init__()
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if not (0.0 <= low_cut_hz < high_cut_hz <= sample_rate / 2.0):
            raise ValueError(
                "Expected 0 <= low_cut_hz < high_cut_hz <= Nyquist; got "
                f"{low_cut_hz}, {high_cut_hz}, sample_rate={sample_rate}"
            )
        self.channels = channels
        self.hidden = hidden
        self.sample_rate = float(sample_rate)
        self.low_cut_hz = float(low_cut_hz)
        self.high_cut_hz = float(high_cut_hz)
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(hidden, hidden, kernel_size=3, stride=2),
        )

        def make_band_branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(channels, 48, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(48),
                nn.GELU(),
                DepthwiseSeparableBlock(48, 72, kernel_size=5, stride=2),
                DepthwiseSeparableBlock(72, hidden, kernel_size=3, stride=2),
            )

        self.frequency_bands = nn.ModuleList([make_band_branch() for _ in range(3)])
        gate_hidden = max(hidden // 2, 16)
        self.band_gate = nn.Sequential(
            nn.Linear(hidden * 3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 3),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def _band_masks(self, length: int, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        frequencies = torch.fft.rfftfreq(length, d=1.0 / self.sample_rate, device=device)
        masks = [
            frequencies < self.low_cut_hz,
            (frequencies >= self.low_cut_hz) & (frequencies <= self.high_cut_hz),
            frequencies > self.high_cut_hz,
        ]
        return [mask.to(dtype=dtype).reshape(1, 1, -1) for mask in masks]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"Expected [batch, time, {self.channels}], got {tuple(x.shape)}")
        x_time = x.permute(0, 2, 1)
        temporal_features = self.temporal(x_time)
        # Reconstruct band-limited time-domain signals rather than feeding
        # differently-sized FFT magnitudes directly to the CNNs.  This keeps
        # the three branches on the same temporal scale and permits a stable
        # per-band normalization (important because the high band typically
        # has much smaller energy than the 0.5--30 Hz band).
        spectrum_complex = torch.fft.rfft(x_time.float(), dim=-1, norm="ortho")
        masks = self._band_masks(x.shape[1], spectrum_complex.device, spectrum_complex.real.dtype)
        band_features = []
        for branch, mask in zip(self.frequency_bands, masks, strict=True):
            band_time = torch.nan_to_num(
                torch.fft.irfft(spectrum_complex * mask, n=x.shape[1], dim=-1, norm="ortho")
            ).to(dtype=x_time.dtype)
            band_time = band_time - band_time.mean(dim=-1, keepdim=True)
            band_time = band_time / band_time.std(dim=-1, keepdim=True).clamp_min(1e-4)
            band_features.append(branch(band_time))
        target_len = min(temporal_features.shape[-1], *(features.shape[-1] for features in band_features))
        temporal_features = temporal_features[..., :target_len]
        band_features = [features[..., :target_len] for features in band_features]
        band_summary = torch.cat([features.mean(dim=-1) for features in band_features], dim=1)
        band_weights = torch.softmax(self.band_gate(band_summary), dim=1)
        frequency_features = sum(
            weight.unsqueeze(1).unsqueeze(-1) * features
            for weight, features in zip(band_weights.unbind(dim=1), band_features, strict=True)
        )
        fused = self.fusion(torch.cat([temporal_features, frequency_features], dim=1))
        sequence = fused.permute(2, 0, 1)
        _, hidden = self.gru(sequence)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class ResidualTriBandDualStreamFreqNet(DualStreamFreqNet):
    """Original DSF plus a zero-initialized three-band residual branch."""

    def __init__(self, channels: int, hidden: int = 96, dropout: float = 0.25, sample_rate: float = 256.0):
        super().__init__(channels=channels, hidden=hidden, dropout=dropout)
        self.sample_rate = float(sample_rate)
        self.low_cut_hz, self.high_cut_hz = 0.5, 30.0
        self.band_aux = nn.Sequential(
            nn.Conv1d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        # The auxiliary branch starts effectively disabled and is bounded to
        # at most 10% of the original spectral representation.  This keeps
        # the residual enhancement from overwhelming the well-tested DSF
        # pathway when the auxiliary bands contain artifacts.
        self.band_residual_scale = nn.Parameter(torch.tensor(-6.0))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.temporal[0].in_channels:
            raise ValueError(f"Expected [batch, time, {self.temporal[0].in_channels}], got {tuple(x.shape)}")
        x_time = x.permute(0, 2, 1)
        temporal_features = self.temporal(x_time)
        spectrum = torch.fft.rfft(x_time, dim=-1)
        full_features = self.frequency(torch.log1p(spectrum.abs()))
        freqs = torch.fft.rfftfreq(x.shape[1], d=1.0 / self.sample_rate, device=x.device)
        masks = [freqs < self.low_cut_hz, (freqs >= self.low_cut_hz) & (freqs <= self.high_cut_hz), freqs > self.high_cut_hz]
        # Avoid feeding the dominant mains components to the auxiliary high
        # band.  The notch is narrow (±1 Hz) and only affects the residual
        # branch; the original DSF spectrum remains untouched.
        mains = ((freqs - 50.0).abs() <= 1.0) | ((freqs - 60.0).abs() <= 1.0)
        masks[2] = masks[2] & ~mains
        bands = [torch.fft.irfft(spectrum * m.reshape(1, 1, -1), n=x.shape[1], dim=-1) for m in masks]
        bands = [b.sub(b.mean(dim=-1, keepdim=True)).div(b.std(dim=-1, keepdim=True).clamp_min(1e-4)) for b in bands]
        target_len = min(temporal_features.shape[-1], full_features.shape[-1])
        residual = self.band_aux(torch.cat(bands, dim=1))[..., :target_len]
        full_features = full_features[..., :target_len]
        residual_scale = 0.1 * torch.sigmoid(self.band_residual_scale)
        # Match the residual scale to the original branch on a per-sample
        # basis, preventing high-amplitude auxiliary activations from causing
        # a distribution shift.
        full_rms = full_features.detach().pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-4)
        residual_rms = residual.detach().pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-4)
        residual = residual * (full_rms / residual_rms) * residual_scale
        fused = self.fusion(torch.cat([temporal_features[..., :target_len], full_features + residual], dim=1))
        _, hidden_state = self.gru(fused.permute(2, 0, 1))
        return torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)


class SameDomainResidualTriBandDSF(DualStreamFreqNet):
    """DSF with same-domain log-spectrum three-band residual features.

    The original full-spectrum DSF path is untouched.  Each masked
    log-magnitude spectrum is encoded in the same convolutional feature space
    and added with a per-band gate initialized exactly at zero, so the initial
    network is functionally identical to DSF.
    """

    def __init__(self, channels: int, hidden: int = 96, dropout: float = 0.25, sample_rate: float = 256.0):
        super().__init__(channels=channels, hidden=hidden, dropout=dropout)
        self.sample_rate = float(sample_rate)
        self.band_residual_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
                    nn.BatchNorm1d(32), nn.GELU(),
                    DepthwiseSeparableBlock(32, 64, kernel_size=5, stride=2),
                    DepthwiseSeparableBlock(64, hidden, kernel_size=3, stride=2),
                )
                for _ in range(3)
            ]
        )
        self.band_residual_gates = nn.Parameter(torch.zeros(3))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.temporal[0].in_channels:
            raise ValueError(f"Expected [batch, time, {self.temporal[0].in_channels}], got {tuple(x.shape)}")
        x_time = x.permute(0, 2, 1)
        temporal_features = self.temporal(x_time)
        spectrum = torch.fft.rfft(x_time, dim=-1)
        full_features = self.frequency(torch.log1p(spectrum.abs()))
        freqs = torch.fft.rfftfreq(x.shape[1], d=1.0 / self.sample_rate, device=x.device)
        masks = [freqs < 0.5, (freqs >= 0.5) & (freqs <= 30.0), freqs > 30.0]
        residuals = []
        for branch, mask in zip(self.band_residual_branches, masks, strict=True):
            band_log = torch.log1p((spectrum * mask.reshape(1, 1, -1)).abs())
            residuals.append(branch(band_log))
        target_len = min(temporal_features.shape[-1], full_features.shape[-1], *(r.shape[-1] for r in residuals))
        temporal_features = temporal_features[..., :target_len]
        full_features = full_features[..., :target_len]
        residual = sum(
            torch.tanh(gate) * band[..., :target_len]
            for gate, band in zip(self.band_residual_gates, residuals, strict=True)
        )
        fused = self.fusion(torch.cat([temporal_features, full_features + residual], dim=1))
        _, hidden_state = self.gru(fused.permute(2, 0, 1))
        return torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class FeatureResidualTriBandDSF(DualStreamFreqNet):
    """DSF with interpretable three-band statistics at the classifier level.

    The original DSF temporal and full-spectrum pathways are untouched. Three
    bands contribute only compact log-energy, relative-energy and spectral
    entropy statistics through a small MLP residual on the final feature
    vector. The residual gate is initialized exactly at zero.
    """

    def __init__(self, channels: int, hidden: int = 96, dropout: float = 0.25, sample_rate: float = 256.0):
        super().__init__(channels=channels, hidden=hidden, dropout=dropout)
        self.sample_rate = float(sample_rate)
        self.band_stat_mlp = nn.Sequential(
            nn.Linear(9, 32), nn.GELU(), nn.LayerNorm(32), nn.Linear(32, hidden * 2)
        )
        self.band_stat_gate = nn.Parameter(torch.tensor(0.0))
        # Keep the gate at zero for exact DSF-equivalent initialization, but
        # retain a small nonzero MLP so the gate receives a useful gradient.
        nn.init.normal_(self.band_stat_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.band_stat_mlp[-1].bias)

    def _band_statistics(self, x_time: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x_time.float(), dim=-1)
        power = spectrum.abs().pow(2)
        freqs = torch.fft.rfftfreq(x_time.shape[-1], d=1.0 / self.sample_rate, device=x_time.device)
        masks = [freqs < 0.5, (freqs >= 0.5) & (freqs <= 30.0), freqs > 30.0]
        total = power.sum(dim=-1).clamp_min(1e-6)
        stats = []
        for mask in masks:
            band_power = power[..., mask]
            energy = band_power.mean(dim=-1)
            ratio = band_power.sum(dim=-1) / total
            prob = band_power / band_power.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1)
            stats.extend([torch.log1p(energy).mean(dim=1), ratio.mean(dim=1), entropy.mean(dim=1)])
        return torch.stack(stats, dim=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        base = super().extract_features(x)
        stats = self._band_statistics(x.permute(0, 2, 1))
        residual = self.band_stat_mlp(stats)
        return base + 0.05 * torch.tanh(self.band_stat_gate) * residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class GraphResidualDSF(DualStreamFreqNet):
    """DSF plus a lightweight DUFGNet-style global/local graph residual."""

    def __init__(self, channels: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__(channels=channels, hidden=hidden, dropout=dropout)
        self.graph_channels = channels
        node_dim = 8
        graph_dim = 32
        self.graph_node = nn.Sequential(nn.Linear(node_dim, graph_dim), nn.GELU())
        self.graph_update1 = nn.Linear(graph_dim, graph_dim, bias=False)
        self.graph_update2 = nn.Linear(graph_dim, graph_dim, bias=False)
        self.graph_global = nn.Sequential(nn.Linear(graph_dim * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden * 2))
        self.graph_gate = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_uniform_(self.graph_update1.weight)
        nn.init.xavier_uniform_(self.graph_update2.weight)
        nn.init.normal_(self.graph_global[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.graph_global[-1].bias)
        adjacency = torch.eye(channels)
        for i in range(channels - 1):
            adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
        # Connect homologous anterior/posterior left-right channel groups when
        # the canonical 17-channel ordering is used.
        for i, j in ((0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)):
            if i < channels and j < channels:
                adjacency[i, j] = adjacency[j, i] = 1.0
        degree = adjacency.sum(dim=1).clamp_min(1.0)
        norm = degree.rsqrt().unsqueeze(1) * adjacency * degree.rsqrt().unsqueeze(0)
        self.register_buffer("graph_adjacency", norm)

    def _graph_features(self, x_time: torch.Tensor) -> torch.Tensor:
        # Compact per-channel temporal/spectral descriptors.
        mean = x_time.mean(dim=-1)
        std = x_time.std(dim=-1).clamp_min(1e-4)
        centered = x_time - mean.unsqueeze(-1)
        spectrum = torch.fft.rfft(centered.float(), dim=-1).abs()
        power = spectrum.pow(2)
        total = power.sum(dim=-1).clamp_min(1e-6)
        freqs = torch.fft.rfftfreq(x_time.shape[-1], d=1.0 / 256.0, device=x_time.device)
        bands = [(freqs < 0.5), ((freqs >= 0.5) & (freqs <= 30.0)), (freqs > 30.0)]
        ratios = [power[..., mask].sum(dim=-1) / total for mask in bands]
        return torch.stack([mean, std.log(), *[r.log1p() for r in ratios], centered.abs().mean(dim=-1), centered.pow(2).mean(dim=-1).sqrt(), spectrum.mean(dim=-1)], dim=-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        base = super().extract_features(x)
        x_time = x.permute(0, 2, 1)
        nodes = self.graph_node(self._graph_features(x_time))
        nodes = torch.relu(torch.einsum("ij,bjd->bid", self.graph_adjacency, self.graph_update1(nodes)))
        nodes = torch.relu(torch.einsum("ij,bjd->bid", self.graph_adjacency, self.graph_update2(nodes)))
        global_node = nodes.mean(dim=1)
        graph_summary = torch.cat([global_node, nodes.amax(dim=1)], dim=-1)
        residual = self.graph_global(graph_summary)
        return base + torch.tanh(self.graph_gate) * residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class DUFGUNetEncoder1d(nn.Module):
    """Three-stage U-Net encoder used by the paper-faithful DUFGNet replica."""

    def __init__(self, in_channels: int):
        super().__init__()
        widths = (16, 32, 64, 128)
        blocks = []
        current = in_channels
        for width in widths:
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(current, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(width),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(width, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(width),
                    nn.ReLU(inplace=True),
                )
            )
            current = width
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.forward_with_skips(x)
        return x

    def forward_with_skips(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)
            if index < len(self.blocks) - 1:
                x = self.pool(x)
        return x, skips


class DUFGUNetDecoder1d(nn.Module):
    """U-Net decoder for the MIDS point-wise three-class pretext task."""

    def __init__(self, output_classes: int = 3):
        super().__init__()
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
                nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
                nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            ]
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv1d(32, 16, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(16),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.head = nn.Conv1d(16, output_classes, kernel_size=1)

    def forward(self, latent: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = latent
        for index, (up, block) in enumerate(zip(self.up, self.blocks, strict=True)):
            x = up(x)
            skip = skips[-2 - index]
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, size=skips[0].shape[-1], mode="linear", align_corners=False)
        return self.head(x)


class DCTChannelAttention1d(nn.Module):
    """Multi-spectral DCT channel attention from DUFGNet.

    Feature channels are split evenly across the first components DCT-II
    bases. The resulting spectral descriptor is mapped to channel weights by
    the two-layer MLP described in the paper.
    """

    def __init__(self, feature_dim: int = 128, components: int = 4, reduction: int = 16):
        super().__init__()
        if feature_dim % components:
            raise ValueError(
                f"feature_dim={feature_dim} must be divisible by components={components}"
            )
        self.feature_dim = feature_dim
        self.components = components
        hidden = max(feature_dim // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def dct_bases(
        length: int,
        components: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32) + 0.5
        frequencies = torch.arange(components, device=device, dtype=torch.float32)
        bases = torch.cos(torch.pi * frequencies[:, None] * positions[None, :] / length)
        bases[0] *= length ** -0.5
        if components > 1:
            bases[1:] *= (2.0 / length) ** 0.5
        return bases.to(dtype=dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 3 or features.shape[-2] != self.feature_dim:
            raise ValueError(
                f"Expected [..., {self.feature_dim}, time], got {tuple(features.shape)}"
            )
        length = features.shape[-1]
        group_width = self.feature_dim // self.components
        grouped = features.reshape(
            *features.shape[:-2],
            self.components,
            group_width,
            length,
        )
        bases = self.dct_bases(
            length,
            self.components,
            device=features.device,
            dtype=features.dtype,
        )
        descriptor = torch.einsum("...kdh,kh->...kd", grouped, bases)
        descriptor = descriptor.reshape(*features.shape[:-2], self.feature_dim)
        weights = self.mlp(descriptor).unsqueeze(-1)
        return features * weights


class DUFGGlobalLocalGraphIntegration(nn.Module):
    """Star-graph attention that updates one global node from channel nodes."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.global_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.local_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(feature_dim, 1),
        )
        nn.init.xavier_uniform_(self.global_projection.weight)
        nn.init.xavier_uniform_(self.local_projection.weight)

    def forward(
        self,
        global_feature: torch.Tensor,
        local_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if global_feature.ndim != 2 or local_features.ndim != 3:
            raise ValueError(
                "Expected global [batch, feature] and local [batch, channels, feature]"
            )
        projected_global = self.global_projection(global_feature)
        projected_local = self.local_projection(local_features)
        expanded_global = projected_global.unsqueeze(1).expand_as(projected_local)
        pairs = torch.cat([expanded_global, projected_local], dim=-1)
        logits = self.attention(pairs).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        integrated = global_feature + torch.sum(
            weights.unsqueeze(-1) * projected_local,
            dim=1,
        )
        return integrated, weights


class DUFGNet(nn.Module):
    """Independent, paper-faithful implementation of DUFGNet.

    A MIDS+TL checkpoint can initialize local_encoder for full DUFGNet while
    preserving the repository's [batch, time, channels] input contract.
    Without that checkpoint this is the paper's S-DUFGNet variant.
    """

    def __init__(
        self,
        channels: int,
        dct_components: int = 4,
        attention_reduction: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.feature_dim = 128
        self.global_encoder = DUFGUNetEncoder1d(in_channels=channels)
        self.local_encoder = DUFGUNetEncoder1d(in_channels=1)
        self.dct_attention = DCTChannelAttention1d(
            feature_dim=self.feature_dim,
            components=dct_components,
            reduction=attention_reduction,
        )
        self.graph_integration = DUFGGlobalLocalGraphIntegration(
            feature_dim=self.feature_dim
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 2, 2),
        )

    def encode_streams(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(
                f"Expected [batch, time, {self.channels}], got {tuple(x.shape)}"
            )
        batch, time_steps, channels = x.shape
        channel_first = x.permute(0, 2, 1)
        global_map = self.global_encoder(channel_first)
        local_input = channel_first.reshape(batch * channels, 1, time_steps)
        local_map = self.local_encoder(local_input)
        local_map = local_map.reshape(
            batch,
            channels,
            self.feature_dim,
            local_map.shape[-1],
        )
        global_map = self.dct_attention(global_map)
        local_map = self.dct_attention(local_map)
        return global_map, local_map

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        global_map, local_map = self.encode_streams(x)
        global_feature = global_map.mean(dim=-1)
        local_features = local_map.mean(dim=-1)
        integrated, _ = self.graph_integration(global_feature, local_features)
        return integrated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class FftEmdDualStreamFreqNet(nn.Module):
    """DSF with an EMD IMF branch in addition to the original FFT branch."""

    def __init__(self, channels: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        if channels <= 12 or (channels - 12) % 3 != 0:
            raise ValueError(
                "fft_emd_dsf expects raw channels followed by three IMF banks "
                f"(total channels = raw + 3*k, with the default 12 IMF channels), got {channels}"
            )
        self.imf_channels = 12
        self.base_channels = channels - self.imf_channels
        self.temporal = nn.Sequential(
            nn.Conv1d(self.base_channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(hidden, hidden, kernel_size=3, stride=2),
        )
        self.fft = nn.Sequential(
            nn.Conv1d(self.base_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=3, stride=2),
        )
        self.emd = nn.Sequential(
            nn.Conv1d(self.imf_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=3, stride=2),
        )
        self.frequency_fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        raw = x[..., : self.base_channels]
        imfs = x[..., self.base_channels :]
        raw_time = raw.permute(0, 2, 1)
        imf_time = imfs.permute(0, 2, 1)
        temporal_features = self.temporal(raw_time)
        fft_features = self.fft(torch.log1p(torch.fft.rfft(raw_time, dim=-1).abs()))
        emd_features = self.emd(imf_time)
        target_len = min(
            temporal_features.shape[-1], fft_features.shape[-1], emd_features.shape[-1]
        )
        frequency_features = self.frequency_fusion(
            torch.cat(
                [fft_features[..., :target_len], emd_features[..., :target_len]], dim=1
            )
        )
        fused = self.fusion(
            torch.cat([temporal_features[..., :target_len], frequency_features], dim=1)
        )
        _, hidden = self.gru(fused.permute(2, 0, 1))
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class AdaptiveSpectralDualStreamFreqNet(nn.Module):
    """DSF with a learnable adaptive spectral filter before the FFT branch."""

    def __init__(self, channels: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.spectral_norm = nn.LayerNorm(channels)
        self.adaptive_spectral = AdaptiveSpectralBlock1d(channels=channels)
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(hidden, hidden, kernel_size=3, stride=2),
        )
        self.frequency = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=3, stride=2),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        x_time = x.permute(0, 2, 1)
        temporal_features = self.temporal(x_time)

        spectrally_filtered = x + self.adaptive_spectral(self.spectral_norm(x))
        freq = torch.fft.rfft(spectrally_filtered.permute(0, 2, 1), dim=-1).abs()
        freq_features = self.frequency(torch.log1p(freq))

        target_len = min(temporal_features.shape[-1], freq_features.shape[-1])
        temporal_features = temporal_features[..., :target_len]
        freq_features = freq_features[..., :target_len]
        fused = self.fusion(torch.cat([temporal_features, freq_features], dim=1))
        sequence = fused.permute(2, 0, 1)
        _, hidden = self.gru(sequence)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class DwtDualStreamFreqNet(nn.Module):
    """Dual-stream temporal/wavelet CNN for seizure prediction.

    This variant keeps the same input/output contract as ``DualStreamFreqNet``
    but replaces the FFT magnitude branch with a lightweight multilevel Haar
    DWT representation implemented in pure PyTorch. The DWT branch exposes
    local multi-scale changes while avoiding extra runtime dependencies.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 96,
        dropout: float = 0.25,
        wavelet_levels: int = 4,
    ):
        super().__init__()
        self.wavelet_levels = wavelet_levels
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(hidden, hidden, kernel_size=3, stride=2),
        )
        self.wavelet = nn.Sequential(
            nn.Conv1d(channels * (wavelet_levels + 1), 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=3, stride=2),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    @staticmethod
    def _haar_step(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] % 2:
            x = nn.functional.pad(x, (0, 1), mode="replicate")
        even = x[..., 0::2]
        odd = x[..., 1::2]
        scale = 2.0 ** -0.5
        approx = (even + odd) * scale
        detail = (even - odd) * scale
        return approx, detail

    def haar_dwt_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return aligned multilevel Haar coefficients as channel features."""

        coeffs = []
        approx = x
        original_len = x.shape[-1]
        for _ in range(self.wavelet_levels):
            approx, detail = self._haar_step(approx)
            coeffs.append(detail.abs())
        coeffs.append(approx.abs())
        aligned = [
            nn.functional.interpolate(c, size=original_len, mode="linear", align_corners=False)
            for c in coeffs
        ]
        return torch.log1p(torch.cat(aligned, dim=1))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x_time = x.permute(0, 2, 1)
        temporal_features = self.temporal(x_time)

        wavelet_input = self.haar_dwt_features(x_time)
        wavelet_features = self.wavelet(wavelet_input)

        target_len = min(temporal_features.shape[-1], wavelet_features.shape[-1])
        temporal_features = temporal_features[..., :target_len]
        wavelet_features = wavelet_features[..., :target_len]
        fused = self.fusion(torch.cat([temporal_features, wavelet_features], dim=1))
        sequence = fused.permute(2, 0, 1)
        _, hidden = self.gru(sequence)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class FixedWaveletDualStreamFreqNet(nn.Module):
    """DSF with a fixed db4 or sym5 DWT frequency branch.

    Unlike the Haar ablation above, this version uses longer, near-symmetric
    analysis filters. The transform remains inside the model so it runs on the
    GPU and preserves the exact window-level data pipeline.
    """

    _ANALYSIS_FILTERS = {
        "db4": (
            [-0.010597401785069032, 0.032883011666982945, 0.030841381835560764, -0.18703481171888114, -0.02798376941698385, 0.6308807679298587, 0.7148465705529154, 0.2303778133088965],
            [-0.2303778133088965, 0.7148465705529154, -0.6308807679298587, -0.02798376941698385, 0.18703481171888114, 0.030841381835560764, -0.032883011666982945, -0.010597401785069032],
        ),
        "sym5": (
            [0.003335725285473771, -0.012580751999015526, -0.006241490212798274, 0.07757149384006515, -0.032244869584638374, -0.24229488706619015, 0.13842814590132074, 0.7243085284385744, 0.6038292697971895, 0.1601023979741929],
            [-0.1601023979741929, 0.6038292697971895, -0.7243085284385744, 0.13842814590132074, 0.24229488706619015, -0.032244869584638374, -0.07757149384006515, -0.006241490212798274, 0.012580751999015526, 0.003335725285473771],
        ),
    }

    def __init__(
        self,
        channels: int,
        wavelet_name: str,
        hidden: int = 96,
        dropout: float = 0.25,
        wavelet_levels: int = 4,
    ):
        super().__init__()
        wavelet_name = wavelet_name.lower()
        if wavelet_name not in self._ANALYSIS_FILTERS:
            raise ValueError(f"Unsupported fixed wavelet: {wavelet_name}")
        low_pass, high_pass = self._ANALYSIS_FILTERS[wavelet_name]
        self.wavelet_name = wavelet_name
        self.wavelet_levels = wavelet_levels
        self.filter_length = len(low_pass)
        self.register_buffer("analysis_low", torch.tensor(low_pass, dtype=torch.float32))
        self.register_buffer("analysis_high", torch.tensor(high_pass, dtype=torch.float32))
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(hidden, hidden, kernel_size=3, stride=2),
        )
        self.wavelet = nn.Sequential(
            nn.Conv1d(channels * (wavelet_levels + 1), 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=3, stride=2),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def _wavelet_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channels = x.shape[1]
        left_pad = self.filter_length // 2 - 1
        right_pad = self.filter_length // 2
        padded = F.pad(x, (left_pad, right_pad), mode="replicate")
        low_filter = self.analysis_low.to(dtype=x.dtype).flip(0).reshape(1, 1, -1).repeat(channels, 1, 1)
        high_filter = self.analysis_high.to(dtype=x.dtype).flip(0).reshape(1, 1, -1).repeat(channels, 1, 1)
        approx = F.conv1d(padded, low_filter, stride=2, groups=channels)
        detail = F.conv1d(padded, high_filter, stride=2, groups=channels)
        return approx, detail

    def wavelet_dwt_features(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = []
        approx = x
        original_len = x.shape[-1]
        for _ in range(self.wavelet_levels):
            approx, detail = self._wavelet_step(approx)
            coeffs.append(detail.abs())
        coeffs.append(approx.abs())
        aligned = [
            F.interpolate(coefficient, size=original_len, mode="linear", align_corners=False)
            for coefficient in coeffs
        ]
        return torch.log1p(torch.cat(aligned, dim=1))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        x_time = x.permute(0, 2, 1)
        temporal_features = self.temporal(x_time)
        wavelet_features = self.wavelet(self.wavelet_dwt_features(x_time))
        target_len = min(temporal_features.shape[-1], wavelet_features.shape[-1])
        fused = self.fusion(
            torch.cat([temporal_features[..., :target_len], wavelet_features[..., :target_len]], dim=1)
        )
        _, hidden = self.gru(fused.permute(2, 0, 1))
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class FftDwtGatedDualStreamFreqNet(nn.Module):
    """DSF with complementary FFT and DWT branches.

    The temporal branch is unchanged from DSF. FFT magnitude gives stable
    whole-window spectral evidence, while the multilevel Haar DWT representation
    preserves localized nonstationary changes. A train-learned softmax gate
    combines their encoded sequences before the final temporal/frequency fusion.
    Checkpoint and alarm-policy selection remain validation-only in the trainer.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 96,
        dropout: float = 0.25,
        wavelet_levels: int = 4,
    ):
        super().__init__()
        self.wavelet_levels = wavelet_levels
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(hidden, hidden, kernel_size=3, stride=2),
        )
        self.fft = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=3, stride=2),
        )
        self.wavelet = nn.Sequential(
            nn.Conv1d(channels * (wavelet_levels + 1), 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=3, stride=2),
        )
        gate_hidden = max(hidden // 2, 16)
        self.frequency_gate = nn.Sequential(
            nn.Conv1d(hidden * 2, gate_hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(gate_hidden),
            nn.GELU(),
            nn.Conv1d(gate_hidden, 2, kernel_size=1),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    @staticmethod
    def _haar_step(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return DwtDualStreamFreqNet._haar_step(x)

    def haar_dwt_features(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = []
        approx = x
        original_len = x.shape[-1]
        for _ in range(self.wavelet_levels):
            approx, detail = self._haar_step(approx)
            coeffs.append(detail.abs())
        coeffs.append(approx.abs())
        aligned = [
            F.interpolate(coefficient, size=original_len, mode="linear", align_corners=False)
            for coefficient in coeffs
        ]
        return torch.log1p(torch.cat(aligned, dim=1))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        x_time = x.permute(0, 2, 1)
        temporal_features = self.temporal(x_time)
        fft_features = self.fft(torch.log1p(torch.fft.rfft(x_time, dim=-1).abs()))
        wavelet_features = self.wavelet(self.haar_dwt_features(x_time))

        target_len = min(temporal_features.shape[-1], fft_features.shape[-1], wavelet_features.shape[-1])
        temporal_features = temporal_features[..., :target_len]
        fft_features = fft_features[..., :target_len]
        wavelet_features = wavelet_features[..., :target_len]
        frequency_weights = torch.softmax(
            self.frequency_gate(torch.cat([fft_features, wavelet_features], dim=1)),
            dim=1,
        )
        frequency_features = (
            frequency_weights[:, :1] * fft_features
            + frequency_weights[:, 1:] * wavelet_features
        )
        fused = self.fusion(torch.cat([temporal_features, frequency_features], dim=1))
        sequence = fused.permute(2, 0, 1)
        _, hidden = self.gru(sequence)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class CwtScalogramDualStreamFreqNet(nn.Module):
    """DSF with a fixed Morlet filter-bank scalogram branch.

    The filter bank is a compact CWT approximation: each EEG channel is
    convolved with several zero-mean, Gaussian-windowed sinusoidal kernels,
    and the absolute responses are encoded as local time-frequency features.
    The filters are fixed; only the downstream classifier is trained.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 96,
        dropout: float = 0.25,
        sample_rate: float = 256.0,
        frequencies: tuple[float, ...] = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0),
        kernel_size: int = 257,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("CWT kernel_size must be odd")
        self.channels = channels
        self.sample_rate = sample_rate
        self.frequencies = tuple(frequencies)
        self.kernel_size = kernel_size
        time = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        time = time / sample_rate
        kernels = []
        for frequency in self.frequencies:
            sigma = 1.0 / (torch.pi * frequency)
            envelope = torch.exp(-0.5 * (time / sigma) ** 2)
            carrier = torch.cos(2.0 * torch.pi * frequency * time)
            kernel = envelope * carrier
            kernel = kernel - kernel.mean()
            kernel = kernel / kernel.norm().clamp_min(1e-8)
            kernels.append(kernel)
        self.register_buffer("cwt_kernels", torch.stack(kernels).unsqueeze(1))
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(hidden, hidden, kernel_size=3, stride=2),
        )
        self.cwt = nn.Sequential(
            nn.Conv1d(channels * len(self.frequencies), 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=3, stride=2),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def scalogram_features(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time_steps = x.shape
        padded = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2), mode="replicate")
        filters = self.cwt_kernels.to(dtype=x.dtype).repeat(channels, 1, 1)
        response = F.conv1d(padded, filters, groups=channels)
        response = response.reshape(batch, channels, len(self.frequencies), time_steps)
        return torch.log1p(response.abs()).reshape(batch, channels * len(self.frequencies), time_steps)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        x_time = x.permute(0, 2, 1)
        temporal_features = self.temporal(x_time)
        cwt_features = self.cwt(self.scalogram_features(x_time))
        target_len = min(temporal_features.shape[-1], cwt_features.shape[-1])
        fused = self.fusion(
            torch.cat([temporal_features[..., :target_len], cwt_features[..., :target_len]], dim=1)
        )
        _, hidden = self.gru(fused.permute(2, 0, 1))
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class LearnableWaveletDualStreamFreqNet(CwtScalogramDualStreamFreqNet):
    """DSF with a trainable, wavelet-initialized local filter bank.

    Filters start from the same Morlet bank as ``CwtScalogramDualStreamFreqNet``
    but are optimized from training windows. Their DC component and energy are
    normalized on every forward pass, keeping a stable band-pass interpretation
    without using any validation or test data to construct the representation.
    """

    def __init__(self, channels: int, **kwargs):
        super().__init__(channels=channels, **kwargs)
        initial_filters = self.cwt_kernels.repeat(channels, 1, 1).clone()
        self.learnable_filters = nn.Parameter(initial_filters)
        del self.cwt_kernels

    def scalogram_features(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time_steps = x.shape
        filters = self.learnable_filters
        filters = filters - filters.mean(dim=-1, keepdim=True)
        filters = filters / filters.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        padded = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2), mode="replicate")
        response = F.conv1d(padded, filters.to(dtype=x.dtype), groups=channels)
        response = response.reshape(batch, channels, len(self.frequencies), time_steps)
        return torch.log1p(response.abs()).reshape(batch, channels * len(self.frequencies), time_steps)


class ChannelAttentionDualStreamFreqNet(nn.Module):
    """DSF variant with shared per-channel temporal encoding.

    The baseline DSF mixes all EEG channels in the first convolution. This
    variant first applies the same temporal encoder to each channel separately,
    then learns a channel-attention weighted fusion. A second shared encoder is
    applied to per-channel FFT magnitudes so the model still keeps the original
    temporal/frequency two-view design.
    """

    def __init__(self, channels: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.channels = channels
        per_channel_hidden = hidden // 2
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            DepthwiseSeparableBlock(16, 32, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(32, per_channel_hidden, kernel_size=5, stride=2),
        )
        self.frequency_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            DepthwiseSeparableBlock(16, 32, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(32, per_channel_hidden, kernel_size=3, stride=2),
        )
        self.channel_attention = nn.Sequential(
            nn.LayerNorm(per_channel_hidden * 2),
            nn.Linear(per_channel_hidden * 2, per_channel_hidden),
            nn.GELU(),
            nn.Linear(per_channel_hidden, 1),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def _encode_per_channel(self, x_ch_first: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        batch, channels, time_steps = x_ch_first.shape
        flat = x_ch_first.reshape(batch * channels, 1, time_steps)
        encoded = encoder(flat)
        return encoded.reshape(batch, channels, encoded.shape[1], encoded.shape[2])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        x_time = x.permute(0, 2, 1)
        temporal = self._encode_per_channel(x_time, self.temporal_encoder)

        freq = torch.fft.rfft(x_time, dim=-1).abs()
        freq = torch.log1p(freq)
        frequency = self._encode_per_channel(freq, self.frequency_encoder)

        target_len = min(temporal.shape[-1], frequency.shape[-1])
        temporal = temporal[..., :target_len]
        frequency = frequency[..., :target_len]
        per_channel = torch.cat([temporal, frequency], dim=2)

        channel_summary = per_channel.mean(dim=-1)
        weights = torch.softmax(self.channel_attention(channel_summary), dim=1)
        fused = (per_channel * weights.unsqueeze(-1)).sum(dim=1)
        fused = self.fusion(fused)
        sequence = fused.permute(2, 0, 1)
        _, hidden = self.gru(sequence)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class ChannelTokenSelfAttentionDualStreamNet(nn.Module):
    """Dual-stream per-channel tokens with pairwise self-attention.

    Unlike ``ChannelAttentionDualStreamFreqNet``, which learns one independent
    scalar weight per channel, this model treats every EEG derivation as a
    token. Multi-head self-attention can therefore model pairwise and
    higher-order channel interactions before window classification.
    """

    def __init__(
        self,
        channels: int,
        token_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.channels = channels
        branch_dim = 48
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            DepthwiseSeparableBlock(16, 32, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(32, branch_dim, kernel_size=5, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.frequency_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            DepthwiseSeparableBlock(16, 32, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(32, branch_dim, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.token_projection = nn.Sequential(
            nn.LayerNorm(branch_dim * 2),
            nn.Linear(branch_dim * 2, token_dim),
            nn.GELU(),
        )
        self.channel_embedding = nn.Parameter(torch.zeros(1, channels, token_dim))
        self.class_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=attention_heads,
            dim_feedforward=token_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=attention_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(token_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(token_dim, 2),
        )
        nn.init.normal_(self.channel_embedding, std=0.02)
        nn.init.normal_(self.class_token, std=0.02)

    @staticmethod
    def _encode_channel_tokens(
        x_ch_first: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        batch, channels, time_steps = x_ch_first.shape
        flat = x_ch_first.reshape(batch * channels, 1, time_steps)
        encoded = encoder(flat).squeeze(-1)
        return encoded.reshape(batch, channels, -1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        x_ch_first = x.permute(0, 2, 1)
        temporal_tokens = self._encode_channel_tokens(
            x_ch_first,
            self.temporal_encoder,
        )

        frequency = torch.log1p(torch.fft.rfft(x_ch_first, dim=-1).abs())
        frequency_tokens = self._encode_channel_tokens(
            frequency,
            self.frequency_encoder,
        )
        tokens = self.token_projection(
            torch.cat([temporal_tokens, frequency_tokens], dim=-1)
        )
        tokens = tokens + self.channel_embedding
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        attended = self.channel_transformer(
            torch.cat([class_token, tokens], dim=1)
        )
        return self.final_norm(attended[:, 0, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class EEGNetStyleDualStreamFreqNet(nn.Module):
    """EEGNet-style temporal/spatial DSF variant.

    This model explicitly separates temporal filtering from spatial channel
    mixing. The time-domain branch uses Conv2D temporal filters followed by a
    depthwise spatial convolution across all EEG channels, similar to EEGNet.
    The frequency branch applies the same idea to log FFT magnitudes.
    """

    def __init__(self, channels: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.channels = channels
        branch_hidden = hidden // 2
        self.temporal_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 65), stride=(1, 2), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(32, branch_hidden, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(branch_hidden),
            nn.GELU(),
        )
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 33), stride=(1, 2), padding=(0, 16), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(32, branch_hidden, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(branch_hidden),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    @staticmethod
    def _to_sequence(features: torch.Tensor) -> torch.Tensor:
        features = features.squeeze(2)
        return features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        x_map = x.permute(0, 2, 1).unsqueeze(1)
        temporal = self._to_sequence(self.temporal_branch(x_map))

        freq = torch.fft.rfft(x.permute(0, 2, 1), dim=-1).abs()
        freq = torch.log1p(freq).unsqueeze(1)
        frequency = self._to_sequence(self.frequency_branch(freq))

        target_len = min(temporal.shape[-1], frequency.shape[-1])
        temporal = temporal[..., :target_len]
        frequency = frequency[..., :target_len]
        fused = self.fusion(torch.cat([temporal, frequency], dim=1))
        sequence = fused.permute(2, 0, 1)
        _, hidden = self.gru(sequence)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class Conv3DDualStreamFreqNet(nn.Module):
    """3D-convolution temporal/frequency DSF variant.

    The input contract stays identical to the other CHB-MIT models:
    ``[batch, time, channels]``. Internally, the model builds a compact
    channel-by-time-frequency volume by pooling both raw EEG and FFT magnitudes
    into aligned temporal bins, then applies lightweight 3D convolutions across
    channel, view, and local time neighborhoods.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 96,
        dropout: float = 0.25,
        volume_bins: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.volume_bins = volume_bins
        self.volume_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 2, 5), padding=(1, 0, 2), bias=False),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=(3, 1, 5), stride=(1, 1, 2), padding=(1, 0, 2), bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=(3, 1, 3), stride=(1, 1, 2), padding=(1, 0, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Conv1d(64 * channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SqueezeExcite1d(hidden),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def _make_volume(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        x_time = x.permute(0, 2, 1)
        temporal = F.adaptive_avg_pool1d(x_time, self.volume_bins)
        freq = torch.fft.rfft(x_time, dim=-1).abs()
        freq = torch.log1p(freq)
        frequency = F.adaptive_avg_pool1d(freq, self.volume_bins)
        volume = torch.stack([temporal, frequency], dim=2)
        return volume.unsqueeze(1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        volume = self._make_volume(x)
        encoded = self.volume_encoder(volume)
        batch, feat, channels, views, steps = encoded.shape
        sequence_features = encoded.reshape(batch, feat * channels * views, steps)
        projected = self.projection(sequence_features)
        sequence = projected.permute(2, 0, 1)
        _, hidden = self.gru(sequence)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class TemporalContextDualStreamNet(nn.Module):
    """Causal sequence model over consecutive 7-second EEG windows.

    Expected input shape is ``[batch, context, time, channels]``. The context
    dimension contains previous windows plus the current target window, and the
    classifier predicts only the current/last window label.
    """

    def __init__(
        self,
        channels: int,
        encoder_hidden: int = 96,
        context_hidden: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder = DualStreamFreqNet(channels=channels, hidden=encoder_hidden, dropout=dropout)
        feature_dim = self.encoder.classifier[0].normalized_shape[0]
        self.context_gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=context_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(context_hidden),
            nn.Dropout(dropout),
            nn.Linear(context_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected [batch, context, time, channels], got shape={tuple(x.shape)}")
        batch, context, time_steps, channels = x.shape
        flat = x.reshape(batch * context, time_steps, channels)
        features = self.encoder.extract_features(flat).reshape(batch, context, -1)
        sequence_out, _ = self.context_gru(features)
        return self.classifier(sequence_out[:, -1, :])


class CausalSelectiveSSMBlock(nn.Module):
    """Lightweight Mamba-style causal selective state-space block.

    Context sequences in this project are short (typically 6-12 windows), so a
    transparent diagonal state recurrence is preferable to a custom CUDA
    kernel. The decay, input gate, and output gate are conditioned on each
    window embedding, while the depthwise convolution is strictly causal.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 2,
        conv_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if conv_kernel < 1:
            raise ValueError("conv_kernel must be positive")
        inner = d_model * expansion
        self.inner = inner
        self.conv_kernel = conv_kernel
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner * 2)
        self.causal_conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=inner,
            bias=True,
        )
        self.delta_proj = nn.Linear(inner, inner)
        self.input_gate_proj = nn.Linear(inner, inner)
        self.output_gate_proj = nn.Linear(inner, inner)
        self.log_decay = nn.Parameter(torch.zeros(inner))
        self.skip = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        delta = F.softplus(self.delta_proj(u)) + 1e-4
        input_gate = torch.tanh(self.input_gate_proj(u))
        output_gate = torch.sigmoid(self.output_gate_proj(u))
        decay_rate = -torch.exp(self.log_decay).view(1, -1)

        state = torch.zeros(
            u.shape[0],
            self.inner,
            dtype=u.dtype,
            device=u.device,
        )
        outputs = []
        for step in range(u.shape[1]):
            decay = torch.exp(delta[:, step, :] * decay_rate)
            candidate = input_gate[:, step, :] * u[:, step, :]
            state = decay * state + (1.0 - decay) * candidate
            outputs.append(
                output_gate[:, step, :] * state
                + self.skip.view(1, -1) * u[:, step, :]
            )
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        projected, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        convolved = self.causal_conv(projected.transpose(1, 2))
        convolved = convolved[..., : x.shape[1]].transpose(1, 2)
        scanned = self.selective_scan(F.silu(convolved))
        mixed = scanned * F.silu(gate)
        return residual + self.dropout(self.out_proj(mixed))


class MambaContextDualStreamNet(nn.Module):
    """Shared DSF encoder followed by causal Mamba-style context modeling.

    Input is ``[batch, context, time, channels]``. Every EEG window is encoded
    independently by the same DSF backbone, then two causal selective SSM
    blocks model only the preceding-to-current window direction. Classification
    uses the last context position, matching online seizure warning deployment.
    """

    def __init__(
        self,
        channels: int,
        encoder_hidden: int = 96,
        context_dim: int = 128,
        context_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder = DualStreamFreqNet(
            channels=channels,
            hidden=encoder_hidden,
            dropout=dropout,
        )
        feature_dim = self.encoder.classifier[0].normalized_shape[0]
        self.context_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, context_dim),
        )
        self.context_blocks = nn.ModuleList(
            [
                CausalSelectiveSSMBlock(
                    d_model=context_dim,
                    expansion=2,
                    conv_kernel=3,
                    dropout=dropout,
                )
                for _ in range(context_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(context_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                f"Expected [batch, context, time, channels], got shape={tuple(x.shape)}"
            )
        batch, context, time_steps, channels = x.shape
        flat = x.reshape(batch * context, time_steps, channels)
        encoded = self.encoder.extract_features(flat).reshape(batch, context, -1)
        sequence = self.context_projection(encoded)
        for block in self.context_blocks:
            sequence = block(sequence)
        return self.final_norm(sequence[:, -1, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class MultiScaleWindowDualStreamNet(nn.Module):
    """Right-aligned 5/10/15-second multi-input seizure predictor.

    Expected input shape is ``[batch, scales, time, channels]``. The three
    scales are padded to the same length, but each scale also carries an
    internal causal mask so the 5-second branch sees only the most recent
    5 seconds, the 10-second branch sees the most recent 10 seconds, and the
    15-second branch sees the full 15-second window.
    """

    def __init__(
        self,
        channels: int,
        scale_steps: tuple[int, int, int] = (1280, 2560, 3840),
        hidden: int = 96,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.scale_steps = scale_steps
        self.encoder = DualStreamFreqNet(channels=channels, hidden=hidden, dropout=dropout)
        feature_dim = self.encoder.classifier[0].normalized_shape[0]
        self.scale_embedding = nn.Parameter(torch.zeros(len(scale_steps), feature_dim))
        self.fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * len(scale_steps)),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * len(scale_steps), feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 2),
        )

    def _mask_scales(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [batch, scales, time, channels], got shape={tuple(x.shape)}")
        if x.shape[1] != len(self.scale_steps):
            raise ValueError(f"Expected {len(self.scale_steps)} scales, got {x.shape[1]}")
        max_steps = x.shape[2]
        masked = x.clone()
        for scale_index, steps in enumerate(self.scale_steps):
            start = max_steps - int(steps)
            if start > 0:
                masked[:, scale_index, :start, :] = 0.0
        return masked

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mask_scales(x)
        batch, scales, time_steps, channels = x.shape
        flat = x.reshape(batch * scales, time_steps, channels)
        features = self.encoder.extract_features(flat).reshape(batch, scales, -1)
        features = features + self.scale_embedding[None, :, :]
        return features.reshape(batch, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(self.extract_features(x))


class DynamicGraphAttentionBlock(nn.Module):
    """Sample-adaptive graph message passing over EEG channel tokens."""

    def __init__(
        self,
        d_model: int,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.prior_strength = nn.Parameter(torch.tensor(1.0))
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        channel_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, features = x.shape
        qkv = self.qkv(self.norm1(x))
        qkv = qkv.reshape(batch, nodes, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        logits = torch.matmul(query, key.transpose(-1, -2))
        logits = logits / (self.head_dim ** 0.5)
        prior_scale = F.softplus(self.prior_strength)
        logits = logits + prior_scale * channel_prior[None, None, :, :]
        attention = torch.softmax(logits, dim=-1)
        message = torch.matmul(attention, value)
        message = message.permute(0, 2, 1, 3).reshape(batch, nodes, features)
        x = x + self.dropout(self.out_proj(message))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attention


class DynamicGraphAttentionDualStreamNet(nn.Module):
    """DSF with a sample-adaptive channel graph attention branch.

    The original DSF remains the global temporal-frequency backbone. A parallel
    graph stream builds one node per bipolar derivation from temporal statistics
    and five frequency-band powers. Edge attention is conditioned on each EEG
    window and receives a soft prior when two derivations share an electrode.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 96,
        node_dim: int = 96,
        graph_layers: int = 2,
        dropout: float = 0.25,
        sampling_rate: float = 256.0,
    ):
        super().__init__()
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.bands = (
            (0.5, 4.0),
            (4.0, 8.0),
            (8.0, 13.0),
            (13.0, 30.0),
            (30.0, 45.0),
        )
        self.encoder = DualStreamFreqNet(
            channels=channels,
            hidden=hidden,
            dropout=dropout,
        )
        global_dim = self.encoder.classifier[0].normalized_shape[0]
        node_input_dim = 3 + len(self.bands)
        self.node_projection = nn.Sequential(
            nn.LayerNorm(node_input_dim),
            nn.Linear(node_input_dim, node_dim),
            nn.GELU(),
        )
        self.graph_blocks = nn.ModuleList(
            [
                DynamicGraphAttentionBlock(
                    d_model=node_dim,
                    heads=4,
                    dropout=dropout,
                )
                for _ in range(graph_layers)
            ]
        )
        self.pool_gate = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, 1),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(global_dim + node_dim),
            nn.Dropout(dropout),
            nn.Linear(global_dim + node_dim, global_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(global_dim, 2),
        )
        self.register_buffer(
            "channel_prior",
            self._build_channel_prior(channels),
        )

    @staticmethod
    def _build_channel_prior(channels: int) -> torch.Tensor:
        channel_names = CHBMIT_COMMON_CHANNELS[:channels]
        electrode_sets = [set(name.split("-")) for name in channel_names]
        prior = torch.zeros(channels, channels, dtype=torch.float32)
        for left in range(channels):
            for right in range(channels):
                if left == right or electrode_sets[left] & electrode_sets[right]:
                    prior[left, right] = 1.0
        return prior

    def channel_node_features(self, x: torch.Tensor) -> torch.Tensor:
        x_ch_first = x.permute(0, 2, 1)
        mean_abs = x_ch_first.abs().mean(dim=-1)
        standard_deviation = x_ch_first.std(dim=-1, unbiased=False)
        line_length = x_ch_first.diff(dim=-1).abs().mean(dim=-1)

        power = torch.fft.rfft(x_ch_first, dim=-1).abs().pow(2)
        frequency_axis = torch.fft.rfftfreq(
            x.shape[1],
            d=1.0 / self.sampling_rate,
            device=x.device,
        )
        band_powers = []
        for low, high in self.bands:
            mask = (frequency_axis >= low) & (frequency_axis < high)
            if not torch.any(mask):
                raise ValueError(f"No FFT bins available for band {low}-{high} Hz")
            band_powers.append(torch.log1p(power[..., mask].mean(dim=-1)))
        return torch.cat(
            [
                mean_abs.unsqueeze(-1),
                standard_deviation.unsqueeze(-1),
                line_length.unsqueeze(-1),
                torch.stack(band_powers, dim=-1),
            ],
            dim=-1,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        global_features = self.encoder.extract_features(x)
        nodes = self.node_projection(self.channel_node_features(x))
        for graph_block in self.graph_blocks:
            nodes, _ = graph_block(nodes, self.channel_prior)
        pool_weights = torch.softmax(self.pool_gate(nodes), dim=1)
        graph_features = (nodes * pool_weights).sum(dim=1)
        return self.fusion(torch.cat([global_features, graph_features], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class GraphFreqNet(nn.Module):
    """Learnable channel-graph and frequency-band backbone.

    The model keeps the standard ``[batch, time, channels]`` input convention.
    It uses a soft channel adjacency matrix to mix bipolar derivations, then
    combines temporal CNN features with log band-power features.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 96,
        dropout: float = 0.25,
        sampling_rate: float = 256.0,
    ):
        super().__init__()
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.register_buffer("identity", torch.eye(channels))
        self.adj_logits = nn.Parameter(torch.zeros(channels, channels))
        self.bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))

        self.temporal = nn.Sequential(
            nn.Conv1d(channels * 2, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            DepthwiseSeparableBlock(64, 96, kernel_size=7, stride=2),
            DepthwiseSeparableBlock(96, hidden, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(hidden, hidden, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.band_encoder = nn.Sequential(
            nn.LayerNorm(channels * len(self.bands)),
            nn.Linear(channels * len(self.bands), hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden * 2, 2)

    def channel_adjacency(self) -> torch.Tensor:
        logits = self.adj_logits + 2.0 * self.identity
        return torch.softmax(logits, dim=-1)

    def graph_mix(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btc,cd->btd", x, self.channel_adjacency())

    def band_power_features(self, x_graph: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels = x_graph.shape
        x_freq = torch.fft.rfft(x_graph.permute(0, 2, 1), dim=-1).abs().pow(2)
        freq_axis = torch.fft.rfftfreq(
            time_steps,
            d=1.0 / self.sampling_rate,
            device=x_graph.device,
        )
        powers = []
        for low, high in self.bands:
            mask = (freq_axis >= low) & (freq_axis < high)
            if not torch.any(mask):
                mask = torch.ones_like(freq_axis, dtype=torch.bool)
            powers.append(x_freq[..., mask].mean(dim=-1))
        band_power = torch.stack(powers, dim=-1)
        return torch.log1p(band_power).reshape(batch, channels * len(self.bands))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        x_graph = self.graph_mix(x)
        temporal_input = torch.cat([x, x_graph], dim=-1).permute(0, 2, 1)
        temporal_features = self.temporal(temporal_input).squeeze(-1)
        band_features = self.band_encoder(self.band_power_features(x_graph))
        return self.fusion(torch.cat([temporal_features, band_features], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class PretrainedBIOTNet(nn.Module):
    """Braindecode BIOT pretrained adapter for CHB-MIT windows.

    The public 18-channel BIOT checkpoints expect 7-second 200 Hz inputs in
    BIOT's canonical bipolar order. CHB-MIT windows in this repository are
    7-second 256 Hz tensors with the local 21-channel order, so this adapter
    reorders the overlapping channels, zero-fills unavailable BIOT channels,
    and resamples to 1400 time points before forwarding through BIOT.
    """

    def __init__(
        self,
        channels: int,
        repo_id: str = "braindecode/biot-pretrained-six-datasets-18chs",
        n_times: int = 1792,
        target_times: int = 1400,
    ):
        super().__init__()
        if channels != len(CHBMIT_COMMON_CHANNELS):
            raise ValueError(
                "PretrainedBIOTNet currently expects the default "
                f"{len(CHBMIT_COMMON_CHANNELS)} CHB-MIT channel order, got {channels}."
            )
        try:
            from braindecode.models import BIOT
        except ImportError as exc:
            raise ImportError(
                "PretrainedBIOTNet requires braindecode. Install it with "
                "`.venv-gpu/bin/python -m pip install braindecode huggingface_hub safetensors`."
            ) from exc

        self.repo_id = repo_id
        self.target_times = target_times
        self.chbmit_to_idx = {name: idx for idx, name in enumerate(CHBMIT_COMMON_CHANNELS)}
        source_indices = []
        missing_mask = []
        for channel in BIOT_18_CHANNEL_ORDER:
            idx = self.chbmit_to_idx.get(channel)
            source_indices.append(-1 if idx is None else idx)
            missing_mask.append(idx is None)
        self.register_buffer("source_indices", torch.tensor(source_indices, dtype=torch.long))
        self.register_buffer("missing_mask", torch.tensor(missing_mask, dtype=torch.bool))
        self.model = BIOT.from_pretrained(
            repo_id,
            n_outputs=2,
            n_chans=len(BIOT_18_CHANNEL_ORDER),
            n_times=target_times,
            sfreq=200,
        )

    def adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        safe_indices = self.source_indices.clamp_min(0)
        x_ch_first = x.permute(0, 2, 1)
        x_biot = x_ch_first.index_select(1, safe_indices)
        if torch.any(self.missing_mask):
            x_biot = x_biot.clone()
            x_biot[:, self.missing_mask, :] = 0.0
        if x_biot.shape[-1] != self.target_times:
            x_biot = F.interpolate(
                x_biot,
                size=self.target_times,
                mode="linear",
                align_corners=False,
            )
        return x_biot

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.adapt_input(x))


class SpatialAttentionBiLSTMNet(nn.Module):
    """Compact convolutional spatial-attention BiLSTM comparator.

    This is a local, reproducible comparator inspired by recent
    convolution/spatial-attention/enhanced-BiLSTM seizure-prediction models.
    It deliberately keeps the repository's ``[batch, time, channels]``
    interface so the data split and event-level protocol remain unchanged.
    """

    def __init__(self, channels: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden, max(hidden // 4, 8), kernel_size=1),
            nn.GELU(),
            nn.Conv1d(max(hidden // 4, 8), hidden, kernel_size=1),
            nn.Sigmoid(),
        )
        self.bilstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        features = self.frontend(x.permute(0, 2, 1))
        features = features * self.channel_attention(features)
        sequence = features.permute(0, 2, 1)
        outputs, _ = self.bilstm(sequence)
        return outputs[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class FaithfulDCSAEBiLSTMNet(nn.Module):
    """Feature-driven DC-SA-EBiLSTM approximation based on Wang et al. (2026).

    The published model consumes 24 descriptors per channel (DB4 five-level
    DWT: A5/D1--D5, each with Higuchi-FD, fuzzy-entropy, RMS and STD).  The
    repository's common loader supplies raw windows, so descriptors are
    computed here on-the-fly and then passed through the channel-centric
    depthwise-convolution, self-attention and enhanced BiLSTM blocks.
    """

    _DB4_LO = (0.23037781, 0.71484657, 0.63088077, -0.02798377,
               -0.18703481, 0.03084138, 0.03288301, -0.01059740)
    _DB4_HI = (-0.01059740, -0.03288301, 0.03084138, 0.18703481,
               -0.02798377, -0.63088077, 0.71484657, -0.23037781)

    def __init__(self, channels: int, embed_dim: int = 64, hidden: int = 32,
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by heads")
        self.channels = channels
        self.embed = nn.Linear(24, embed_dim)
        self.positional = nn.Parameter(torch.zeros(1, channels, embed_dim))
        nn.init.normal_(self.positional, std=0.02)
        self.dw = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1,
                            groups=embed_dim, bias=False)
        self.pw = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.local_dropout = nn.Dropout(dropout)
        self.local_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout,
                                          batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim))
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden, num_layers=1, batch_first=True,
                              bidirectional=True)
        self.proj = nn.Linear(embed_dim, hidden * 2)
        self.gate = nn.Linear(hidden * 4, hidden * 2)
        self.fuse_norm = nn.LayerNorm(hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(channels * hidden * 2, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 2)
        )
        lo = torch.tensor(self._DB4_LO, dtype=torch.float32).view(1, 1, -1)
        hi = torch.tensor(self._DB4_HI, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("_db4_lo", lo)
        self.register_buffer("_db4_hi", hi)
        # The paper standardizes the handcrafted descriptor vector using the
        # training fold.  These buffers are fitted by the LOOCV trainer after
        # the model is moved to its target device; keeping them in the module
        # makes the transform part of the saved checkpoint and prevents test
        # statistics from leaking into training.
        self.register_buffer("feature_mean", torch.zeros(1, channels, 24))
        self.register_buffer("feature_std", torch.ones(1, channels, 24))
        self.register_buffer("feature_standardized", torch.tensor(False))

    def _raw_dwt_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] -> six subbands [B,C,L]. Fixed DB4 analysis filters.
        # The symmetric three-left/four-right extension is the compact
        # decimated-DWT convention used here.  It keeps each level at half
        # the previous length and is stable for the one-second (256-sample)
        # windows used by the published protocol.
        z = x.float().permute(0, 2, 1)
        # Paper preprocessing: fourth-order 0.5-Hz high-pass plus a narrow
        # 60-Hz notch.  Apply the corresponding zero-phase response before
        # the DWT.  At the published 256-Hz/1-s resolution this removes DC,
        # attenuates the sub-Hz edge, and suppresses the 60-Hz FFT bin while
        # preserving the D1/D2 bands that a 30-Hz low-pass would discard.
        spectrum = torch.fft.rfft(z, dim=-1)
        freq = torch.fft.rfftfreq(z.shape[-1], d=1.0 / 256.0).to(z.device)
        safe_freq = freq.clamp_min(1e-6)
        highpass = torch.rsqrt(1.0 + (0.5 / safe_freq).pow(8))
        highpass[0] = 0.0
        notch = ((freq < 59.0) | (freq > 61.0)).to(highpass.dtype)
        z = torch.fft.irfft(spectrum * (highpass * notch), n=z.shape[-1], dim=-1)
        details = []
        approx = z
        for _ in range(5):
            c = approx.shape[1]
            lo = self._db4_lo.expand(c, 1, -1)
            hi = self._db4_hi.expand(c, 1, -1)
            src = F.pad(approx, (3, 4), mode="reflect")
            next_approx = F.conv1d(src, lo, stride=2, groups=c)
            detail = F.conv1d(src, hi, stride=2, groups=c)
            details.append(detail)
            approx = next_approx
        # Match the paper's stated A5, D1--D5 descriptor order.
        bands = [approx, *details]
        feats = []
        for b in bands:
            std = b.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
            rms = torch.sqrt((b * b).mean(dim=-1, keepdim=True) + 1e-6)
            # Stable differentiable proxies for Higuchi FD and fuzzy entropy.
            d = (b[..., 1:] - b[..., :-1]).abs().mean(dim=-1, keepdim=True)
            hfd = torch.log1p(std / (d + 1e-6))
            tol = 0.2 * std
            fuzzy = -torch.log(torch.exp(-d / (tol + 1e-6)) + 1e-6)
            feats.extend((hfd, fuzzy, rms, std))
        return torch.cat(feats, dim=-1)

    @torch.no_grad()
    def fit_feature_standardizer(self, x: torch.Tensor, batch_size: int = 256) -> None:
        """Fit descriptor mean/std on *training-fold* windows only."""
        # The exact-feature cache supplies [N, channels, 24] directly.  Keep
        # the same fold-local standardization semantics while bypassing DWT.
        if x.ndim == 3 and x.shape[-1] == 24 and x.shape[1] == self.channels:
            descriptors = x.float()
            mean = descriptors.mean(dim=0, keepdim=True)
            variance = descriptors.var(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            self.feature_mean.copy_(mean)
            self.feature_std.copy_(variance.sqrt())
            self.feature_standardized.fill_(True)
            return
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"Expected [N,time,{self.channels}], got {tuple(x.shape)}")
        sums = torch.zeros_like(self.feature_mean)
        squares = torch.zeros_like(self.feature_mean)
        count = 0
        for start in range(0, x.shape[0], max(int(batch_size), 1)):
            descriptors = self._raw_dwt_features(x[start : start + batch_size])
            sums += descriptors.sum(dim=0, keepdim=True)
            squares += descriptors.square().sum(dim=0, keepdim=True)
            count += descriptors.shape[0]
        mean = sums / max(count, 1)
        variance = (squares / max(count, 1) - mean.square()).clamp_min(1e-6)
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(variance.sqrt())
        self.feature_standardized.fill_(True)

    def _dwt_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[-1] == 24 and x.shape[1] == self.channels:
            descriptors = x.float()
            if bool(self.feature_standardized.item()):
                descriptors = (descriptors - self.feature_mean) / self.feature_std.clamp_min(1e-6)
            return descriptors
        descriptors = self._raw_dwt_features(x)
        if bool(self.feature_standardized.item()):
            descriptors = (descriptors - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        return descriptors

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Raw windows use [batch,time,channels]; exact handcrafted-cache
        # inputs use [batch,channels,24].  Both represent the same channel
        # axis and are standardized through _dwt_features below.
        is_descriptor_cache = x.ndim == 3 and x.shape[1] == self.channels and x.shape[-1] == 24
        if x.ndim != 3 or (not is_descriptor_cache and x.shape[-1] != self.channels):
            raise ValueError(
                f"Expected [batch,time,{self.channels}] or [batch,{self.channels},24], got {tuple(x.shape)}"
            )
        h = self.embed(self._dwt_features(x))
        residual = h
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        h = self.pw(h.transpose(1, 2)).transpose(1, 2)
        h = self.local_dropout(h)
        h = self.local_norm(h + residual)
        h = self.attn_norm(h + self.attn(h + self.positional, h + self.positional,
                                         h + self.positional, need_weights=False)[0])
        h = self.ffn_norm(h + self.ffn(h))
        rnn, _ = self.bilstm(h)
        proj = self.proj(h)
        gate = torch.sigmoid(self.gate(torch.cat([rnn, proj], dim=-1)))
        return self.fuse_norm(gate * rnn + (1.0 - gate) * proj).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class TransformerRNNFusionNet(nn.Module):
    """Transformer-RNN fusion comparator for window-level risk scoring."""

    def __init__(self, channels: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=9, stride=8, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden + hidden * 2, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden * 2, 2)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        sequence = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        transformer_features = self.transformer(sequence).mean(dim=1)
        _, hidden = self.gru(sequence)
        rnn_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fusion(torch.cat([transformer_features, rnn_features], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class MaskedReconstructionPretrainer(nn.Module):
    """Self-supervised wrapper for masked EEG window reconstruction."""

    def __init__(self, encoder: DualStreamFreqNet, time_steps: int, channels: int):
        super().__init__()
        self.encoder = encoder
        feature_dim = encoder.classifier[0].normalized_shape[0]
        self.decoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, time_steps * channels),
        )
        self.time_steps = time_steps
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.extract_features(x)
        return self.decoder(features).view(-1, self.time_steps, self.channels)


class FeatureResidualMLP(nn.Module):
    """Identity-initialized residual MLP used as one growth atom."""

    def __init__(self, feature_dim: int, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, feature_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.activation(self.down(self.norm(x))))


class FeatureTemporalResidualBlock(nn.Module):
    """Local mixing over adjacent dimensions of the pooled DSF embedding."""

    def __init__(self, feature_dim: int, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.local = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, feature_dim),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        local = self.local(normalized.unsqueeze(1)).squeeze(1)
        return x + self.projection(local)


class FeatureFrequencyResidualBlock(nn.Module):
    """Learnable spectral mixing over the DSF feature vector."""

    def __init__(self, feature_dim: int, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        frequency_bins = feature_dim // 2 + 1
        # Start as an all-pass transform so the residual projection receives
        # a nonzero signal while the atom is still harmless at initialization.
        initial_gain = torch.zeros(frequency_bins, 2)
        initial_gain[:, 0] = 1.0
        self.complex_gain = nn.Parameter(initial_gain)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, feature_dim),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        spectrum = torch.fft.rfft(normalized, dim=-1)
        gain = torch.view_as_complex(self.complex_gain)
        spectral_branch = torch.fft.irfft(spectrum * gain, n=x.shape[-1], dim=-1)
        return x + self.projection(spectral_branch)


class FeatureChannelGate(nn.Module):
    """Identity-initialized feature-wise gate."""

    def __init__(self, feature_dim: int, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, feature_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered_gate = torch.tanh(self.gate(self.norm(x)))
        return x * (1.0 + centered_gate)


class FeatureTokenAttention(nn.Module):
    """Self-attention over fixed chunks of the pooled embedding."""

    def __init__(self, feature_dim: int, token_count: int = 8, heads: int = 4):
        super().__init__()
        if feature_dim % token_count != 0:
            raise ValueError("feature_dim must be divisible by token_count")
        token_dim = feature_dim // token_count
        if token_dim % heads != 0:
            raise ValueError("token_dim must be divisible by attention heads")
        self.feature_dim = feature_dim
        self.token_count = token_count
        self.token_dim = token_dim
        self.norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        nn.init.zeros_(self.feed_forward[-1].weight)
        nn.init.zeros_(self.feed_forward[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.reshape(x.shape[0], self.token_count, self.token_dim)
        attended, _ = self.attention(self.norm(tokens), self.norm(tokens), self.norm(tokens))
        tokens = tokens + attended
        tokens = tokens + self.feed_forward(tokens)
        return tokens.reshape(x.shape[0], self.feature_dim)


def _progressive_atom(name: str, feature_dim: int) -> nn.Module:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "temporal": "temporal",
        "temporal_residual": "temporal",
        "frequency": "frequency",
        "frequency_residual": "frequency",
        "channel": "channel",
        "channel_gate": "channel",
        "attention": "attention",
        "token_attention": "attention",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "temporal":
        return FeatureTemporalResidualBlock(feature_dim)
    if normalized == "frequency":
        return FeatureFrequencyResidualBlock(feature_dim)
    if normalized == "channel":
        return FeatureChannelGate(feature_dim)
    if normalized == "attention":
        return FeatureTokenAttention(feature_dim)
    raise ValueError(
        f"Unsupported progressive atom {name!r}; use temporal, frequency, channel, or attention"
    )


class ProgressiveDSFNet(nn.Module):
    """DSF with an explicitly ordered list of identity-initialized atoms.

    ``progressive_blocks`` is selected outside the held-out test set. The
    underlying DSF encoder and classifier remain unchanged, while each atom
    adds one trainable residual transformation to the 192-D window embedding.
    """

    def __init__(
        self,
        channels: int,
        progressive_blocks: tuple[str, ...] = (),
        hidden: int = 96,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.progressive_blocks = tuple(
            block.strip().lower().replace("-", "_")
            for block in progressive_blocks
            if block.strip()
        )
        self.encoder = DualStreamFreqNet(channels=channels, hidden=hidden, dropout=dropout)
        feature_dim = self.encoder.classifier[0].normalized_shape[0]
        self.atoms = nn.ModuleList(
            [_progressive_atom(block, feature_dim) for block in self.progressive_blocks]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.extract_features(x)
        for atom in self.atoms:
            features = atom(features)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class InternalTemporalGrowthBlock(nn.Module):
    """Zero-output residual block for temporal feature-map growth."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.activation = nn.GELU()
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.pointwise(self.activation(self.depthwise(self.norm(x))))
        return x + branch


class InternalFrequencyGrowthBlock(InternalTemporalGrowthBlock):
    """Frequency-branch residual atom with a different receptive field."""

    def __init__(self, channels: int):
        super().__init__(channels=channels, kernel_size=5)


class ProgressiveInternalDSFNet(nn.Module):
    """DSF assembled by growing its internal temporal/frequency modules.

    Stages are cumulative and keep a 192-D feature/classifier contract:

    0: temporal stem + pooled classifier
    1: one temporal residual block
    2: FFT branch
    3: one frequency residual block
    4: temporal-frequency fusion
    5: bidirectional GRU sequence module

    Architecture selection is performed outside this class using validation
    results. New branches use zero residual scales or identity-compatible
    projections so inherited checkpoints remain meaningful.
    """

    max_stage = 5

    def __init__(
        self,
        channels: int,
        growth_stage: int = 0,
        hidden: int = 96,
        dropout: float = 0.25,
    ):
        super().__init__()
        if growth_stage < 0 or growth_stage > self.max_stage:
            raise ValueError(f"growth_stage must be in [0, {self.max_stage}], got {growth_stage}")
        self.growth_stage = int(growth_stage)
        self.hidden = hidden
        self.feature_dim = hidden * 2
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.temporal_blocks = nn.ModuleList()
        if self.growth_stage >= 1:
            self.temporal_blocks.append(InternalTemporalGrowthBlock(hidden, kernel_size=7))

        self.temporal_projection = nn.Linear(hidden, self.feature_dim)
        self.frequency_stem = None
        self.frequency_blocks = nn.ModuleList()
        self.frequency_projection = None
        self.frequency_scale = None
        if self.growth_stage >= 2:
            self.frequency_stem = nn.Sequential(
                nn.Conv1d(channels, hidden, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            )
            if self.growth_stage >= 3:
                self.frequency_blocks.append(InternalFrequencyGrowthBlock(hidden))
            self.frequency_projection = nn.Linear(hidden, self.feature_dim)
            self.frequency_scale = nn.Parameter(torch.tensor(0.0))

        self.fusion = None
        if self.growth_stage >= 4:
            self.fusion = nn.Linear(self.feature_dim * 2, self.feature_dim)
            with torch.no_grad():
                self.fusion.weight.zero_()
                self.fusion.bias.zero_()
                self.fusion.weight[:, : self.feature_dim].copy_(torch.eye(self.feature_dim))

        self.gru = None
        self.sequence_projection = None
        self.gru_scale = None
        if self.growth_stage >= 5:
            self.sequence_projection = nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False)
            with torch.no_grad():
                self.sequence_projection.weight.zero_()
                self.sequence_projection.weight[:, :hidden, 0] = torch.eye(hidden)
            self.gru = nn.GRU(hidden, hidden, batch_first=False, bidirectional=True)
            self.gru_scale = nn.Parameter(torch.tensor(0.0))

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 2),
        )

    @staticmethod
    def _pool(feature_map: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        return projection(feature_map.mean(dim=-1))

    def _temporal_map(self, x_time: torch.Tensor) -> torch.Tensor:
        feature_map = self.temporal_stem(x_time)
        for block in self.temporal_blocks:
            feature_map = block(feature_map)
        return feature_map

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, channels], got shape={tuple(x.shape)}")
        x_time = x.permute(0, 2, 1)
        temporal_map = self._temporal_map(x_time)
        temporal_feature = self._pool(temporal_map, self.temporal_projection)
        feature = temporal_feature

        frequency_map = None
        frequency_feature = None
        if self.growth_stage >= 2:
            frequency_input = torch.log1p(torch.fft.rfft(x_time, dim=-1).abs())
            frequency_map = self.frequency_stem(frequency_input)
            for block in self.frequency_blocks:
                frequency_map = block(frequency_map)
            frequency_feature = self._pool(frequency_map, self.frequency_projection)
            if self.growth_stage < 4:
                feature = feature + self.frequency_scale * frequency_feature
            else:
                feature = self.fusion(torch.cat([temporal_feature, frequency_feature], dim=-1))

        if self.growth_stage >= 5:
            if frequency_map is None:
                sequence_map = temporal_map
            else:
                target_len = min(temporal_map.shape[-1], frequency_map.shape[-1])
                sequence_map = torch.cat(
                    [temporal_map[..., :target_len], frequency_map[..., :target_len]],
                    dim=1,
                )
            sequence_map = self.sequence_projection(sequence_map)
            sequence, _ = self.gru(sequence_map.permute(2, 0, 1))
            recurrent_feature = sequence[-1]
            feature = feature + self.gru_scale * recurrent_feature
        return feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def build_model(
    name: str,
    channels: int,
    progressive_blocks: tuple[str, ...] | list[str] | None = None,
    internal_stage: int = 0,
) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized in {"progressive_dsf", "growing_dsf", "dynamic_growth_dsf"}:
        blocks = tuple(progressive_blocks or ())
        return ProgressiveDSFNet(channels=channels, progressive_blocks=blocks)
    if normalized in {
        "progressive_internal_dsf",
        "internal_progressive_dsf",
        "growing_internal_dsf",
    }:
        return ProgressiveInternalDSFNet(channels=channels, growth_stage=internal_stage)
    if normalized in {"cnn_gru", "cnngru", "baseline"}:
        return CNNGRU(channels=channels)
    if normalized in {"dual_stream_freq", "dualstreamfreq", "dsf"}:
        return DualStreamFreqNet(channels=channels)
    if normalized in {"attnsleep", "attn_sleep", "attnsleep_dsf", "attention_sleep"}:
        return AttnSleepNet(channels=channels)
    if normalized in {"tri_band_dsf", "triband_dsf", "three_band_dsf", "3band_dsf"}:
        return TriBandDualStreamFreqNet(channels=channels)
    if normalized in {"residual_tri_band_dsf", "res_tri_band_dsf", "tri_band_residual_dsf"}:
        return ResidualTriBandDualStreamFreqNet(channels=channels)
    if normalized in {"same_domain_residual_tri_band_dsf", "sd_residual_tri_band_dsf", "tri_band_same_domain_dsf"}:
        return SameDomainResidualTriBandDSF(channels=channels)
    if normalized in {"feature_residual_tri_band_dsf", "classifier_residual_tri_band_dsf", "stats_residual_dsf"}:
        return FeatureResidualTriBandDSF(channels=channels)
    if normalized in {"graph_residual_dsf", "dufg_graph_residual_dsf", "global_local_graph_dsf"}:
        return GraphResidualDSF(channels=channels)
    if normalized in {"dufgnet", "s_dufgnet", "dufgnet_scratch"}:
        return DUFGNet(channels=channels)
    if normalized in {"fft_emd_dsf", "emd_dsf", "fft_emd_dual_stream"}:
        return FftEmdDualStreamFreqNet(channels=channels)
    if normalized in {"adaptive_spectral_dsf", "asb_dsf", "tslanet_dsf"}:
        return AdaptiveSpectralDualStreamFreqNet(channels=channels)
    if normalized in {"channel_attention_dsf", "channel_attention_dual_stream", "ca_dsf"}:
        return ChannelAttentionDualStreamFreqNet(channels=channels)
    if normalized in {
        "channel_token_self_attention_dsf",
        "channel_token_attention_dsf",
        "ctsa_dsf",
    }:
        return ChannelTokenSelfAttentionDualStreamNet(channels=channels)
    if normalized in {"eegnet_dsf", "eegnet_dual_stream", "eegnet_style_dsf"}:
        return EEGNetStyleDualStreamFreqNet(channels=channels)
    if normalized in {"conv3d_dsf", "conv3d_dual_stream", "conv3d_dual_stream_freq", "3d_dsf"}:
        return Conv3DDualStreamFreqNet(channels=channels)
    if normalized in {"dwt_dual_stream", "dwt_dual_stream_freq", "dwt_dsf", "wavelet_dsf"}:
        return DwtDualStreamFreqNet(channels=channels)
    if normalized in {"dwt_db4_dsf", "db4_dsf", "db4_wavelet_dsf"}:
        return FixedWaveletDualStreamFreqNet(channels=channels, wavelet_name="db4")
    if normalized in {"dwt_sym5_dsf", "sym5_dsf", "sym5_wavelet_dsf"}:
        return FixedWaveletDualStreamFreqNet(channels=channels, wavelet_name="sym5")
    if normalized in {"cwt_scalogram_dsf", "cwt_dsf", "scalogram_dsf"}:
        return CwtScalogramDualStreamFreqNet(channels=channels)
    if normalized in {"learnable_wavelet_dsf", "learnable_cwt_dsf", "learned_wavelet_dsf"}:
        return LearnableWaveletDualStreamFreqNet(channels=channels)
    if normalized in {"fft_dwt_gated_dsf", "fft_dwt_dsf", "hybrid_wavelet_dsf"}:
        return FftDwtGatedDualStreamFreqNet(channels=channels)
    if normalized in {"temporal_context_dual_stream", "temporal_context_dsf", "context_dsf"}:
        return TemporalContextDualStreamNet(channels=channels)
    if normalized in {"mamba_context_dsf", "mamba_context_dual_stream", "causal_ssm_dsf"}:
        return MambaContextDualStreamNet(channels=channels)
    if normalized in {"multi_scale_window_dsf", "multiscale_window_dsf", "multi_scale_dsf"}:
        return MultiScaleWindowDualStreamNet(channels=channels)
    if normalized in {
        "dynamic_graph_attention_dsf",
        "dynamic_graph_dsf",
        "dga_dsf",
    }:
        return DynamicGraphAttentionDualStreamNet(channels=channels)
    if normalized in {"graph_freq", "graph_frequency", "dufg_lite", "dufgnet_lite"}:
        return GraphFreqNet(channels=channels)
    if normalized in {"pretrained_biot", "biot_pretrained", "biot"}:
        return PretrainedBIOTNet(channels=channels)
    if normalized in {
        "dc_sa_ebilstm",
        "spatial_attention_bilstm",
        "conv_sa_bilstm",
    }:
        return SpatialAttentionBiLSTMNet(channels=channels)
    if normalized in {
        "dc_sa_ebilstm_faithful",
        "faithful_dc_sa_ebilstm",
        "dcsa_ebilstm_faithful",
    }:
        return FaithfulDCSAEBiLSTMNet(channels=channels)
    if normalized in {
        "transformer_rnn_fusion",
        "transformer_rnn",
        "trf_rnn",
    }:
        return TransformerRNNFusionNet(channels=channels)
    raise ValueError(f"Unknown model: {name}")
