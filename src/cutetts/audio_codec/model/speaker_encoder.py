# Copyright 2026 OPPO and Fudan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: Optional[float],
) -> torch.Tensor:
    f_max = float(sample_rate // 2 if f_max is None else f_max)
    all_freqs = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1)
    m_min = _hz_to_mel(torch.tensor(float(f_min)))
    m_max = _hz_to_mel(torch.tensor(f_max))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts)

    filters = []
    for idx in range(n_mels):
        left, center, right = f_pts[idx], f_pts[idx + 1], f_pts[idx + 2]
        up = (all_freqs - left) / (center - left).clamp_min(1e-8)
        down = (right - all_freqs) / (right - center).clamp_min(1e-8)
        filters.append(torch.maximum(torch.zeros_like(all_freqs), torch.minimum(up, down)))
    return torch.stack(filters, dim=0)


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = f_max
        self.eps = float(eps)
        self.register_buffer(
            "mel_filter",
            _mel_filter_bank(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
            ),
            persistent=False,
        )
        self.register_buffer(
            "window",
            torch.hann_window(self.win_length),
            persistent=False,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 3:
            if wav.shape[1] != 1:
                wav = wav.mean(dim=1)
            else:
                wav = wav[:, 0]
        if wav.ndim != 2:
            raise ValueError(f"Expected wav shape [B, T] or [B, C, T], got {tuple(wav.shape)}")

        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=wav.device, dtype=wav.dtype),
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel = torch.matmul(self.mel_filter.to(device=wav.device, dtype=wav.dtype), power)
        return torch.log(mel.clamp_min(self.eps))


class Res2Conv1dReluBn(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        scale: int = 8,
    ):
        super().__init__()
        if channels % scale != 0:
            raise ValueError(f"channels={channels} must be divisible by scale={scale}.")
        self.scale = int(scale)
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        splits = torch.split(x, self.width, dim=1)
        for idx in range(self.nums):
            chunk = splits[idx] if idx == 0 else chunk + splits[idx]
            chunk = self.convs[idx](chunk)
            chunk = self.bns[idx](F.relu(chunk))
            out.append(chunk)
        if self.scale != 1:
            out.append(splits[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(self.conv(x)))


class SEConnect(nn.Module):
    def __init__(self, channels: int, bottleneck_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(channels, bottleneck_dim)
        self.linear2 = nn.Linear(bottleneck_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SERes2Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        scale: int,
        se_bottleneck_dim: int,
    ):
        super().__init__()
        self.conv1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1)
        self.res2 = Res2Conv1dReluBn(
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            scale=scale,
        )
        self.conv2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1)
        self.se = SEConnect(out_channels, se_bottleneck_dim)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.conv1(x)
        x = self.res2(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(
        self,
        in_dim: int,
        attention_channels: int = 128,
        global_context_att: bool = False,
    ):
        super().__init__()
        self.global_context_att = bool(global_context_att)
        input_dim = in_dim * 3 if self.global_context_att else in_dim
        self.linear1 = nn.Conv1d(input_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class FbankECAPAStudent(nn.Module):

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        channels: int = 704,
        emb_dim: int = 256,
        scale: int = 8,
        se_bottleneck_dim: int = 128,
        attention_channels: int = 128,
        global_context_att: bool = False,
        n_fft: int = 512,
        win_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        win_length = int(round(sample_rate * win_length_ms / 1000.0))
        hop_length = int(round(sample_rate * hop_length_ms / 1000.0))
        self.config = {
            "sample_rate": int(sample_rate),
            "n_mels": int(n_mels),
            "channels": int(channels),
            "emb_dim": int(emb_dim),
            "scale": int(scale),
            "se_bottleneck_dim": int(se_bottleneck_dim),
            "attention_channels": int(attention_channels),
            "global_context_att": bool(global_context_att),
            "n_fft": int(n_fft),
            "win_length_ms": float(win_length_ms),
            "hop_length_ms": float(hop_length_ms),
            "eps": float(eps),
        }
        self.sample_rate = int(sample_rate)
        self.emb_dim = int(emb_dim)
        self.feature_extract = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            eps=eps,
        )
        self.instance_norm = nn.InstanceNorm1d(n_mels)
        self.layer1 = Conv1dReluBn(n_mels, channels, kernel_size=5, padding=2)
        self.layer2 = SERes2Block(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            scale=scale,
            se_bottleneck_dim=se_bottleneck_dim,
        )
        self.layer3 = SERes2Block(
            channels,
            channels,
            kernel_size=3,
            padding=3,
            dilation=3,
            scale=scale,
            se_bottleneck_dim=se_bottleneck_dim,
        )
        self.layer4 = SERes2Block(
            channels,
            channels,
            kernel_size=3,
            padding=4,
            dilation=4,
            scale=scale,
            se_bottleneck_dim=se_bottleneck_dim,
        )
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        self.pooling = AttentiveStatsPool(
            cat_channels,
            attention_channels=attention_channels,
            global_context_att=global_context_att,
        )
        self.bn = nn.BatchNorm1d(cat_channels * 2)
        self.linear = nn.Linear(cat_channels * 2, emb_dim)

    def get_config(self):
        return dict(self.config)

    def forward(self, wav: torch.Tensor, sample_rate: Optional[int] = None) -> dict:
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            raise ValueError(f"Expected sample_rate={self.sample_rate}, got {sample_rate}.")
        x = self.feature_extract(wav)
        x = self.instance_norm(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.pooling(out)
        if out.shape[0] == 1:
            out = F.batch_norm(
                out,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                False,
                self.bn.momentum,
                self.bn.eps,
            )
        else:
            out = self.bn(out)
        embedding_raw = self.linear(out)
        embedding = F.normalize(embedding_raw, dim=-1)
        return {
            "embedding": embedding,
            "embedding_raw": embedding_raw,
        }
