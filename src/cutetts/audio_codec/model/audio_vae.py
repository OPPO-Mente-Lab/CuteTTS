# Portions of this file are adapted from Descript Audio Codec (DAC):
# https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/layers.py
# https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/dac.py
#
# Copyright (c) 2023-present, Descript
# Licensed under the MIT License. See LICENSES/DAC-MIT.txt.
#
# Modifications Copyright 2026 OPPO and Fudan University
# Adapted for causal audio encoding/decoding, variational latent modeling,
# and integration with CuteTTS.
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
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from cutetts.audio_codec.model.vae import VAEEncoderOutput

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding

    def forward(self, x):
        x_pad = F.pad(x, (self.__padding * 2, 0))
        return super().forward(x_pad)


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        trim = self.__padding * 2 - self.__output_padding
        y = super().forward(x)
        return y if trim == 0 else y[..., :-trim]


def WNCausalConv1d(*args, **kwargs):
    return weight_norm(CausalConv1d(*args, **kwargs))


def WNCausalTransposeConv1d(*args, **kwargs):
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def init_weights(module, std: float = 2.0e-2):
    if isinstance(module, nn.Conv1d):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class CausalResidualUnit(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        dilation: int = 1,
        kernel: int = 7,
        groups: int = 1,
    ):
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNCausalConv1d(
                dim,
                dim,
                kernel_size=kernel,
                dilation=dilation,
                padding=pad,
                groups=groups,
            ),
            Snake1d(dim),
            WNCausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class CausalEncoderBlock(nn.Module):
    def __init__(
        self,
        output_dim: int = 16,
        input_dim=None,
        stride: int = 1,
        groups=1,
    ):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNCausalConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        latent_dim: int = 32,
        strides: list = None,
        depthwise: bool = False,
    ):
        super().__init__()
        strides = [2, 4, 8, 8] if strides is None else list(strides)
        self.block = [WNCausalConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            self.block += [
                CausalEncoderBlock(
                    output_dim=d_model,
                    stride=stride,
                    groups=groups,
                )
            ]
        self.fc_mu = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = WNCausalConv1d(
            d_model,
            latent_dim,
            kernel_size=3,
            padding=1,
        )
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x, include_logvar: bool = True):
        hidden_state = self.block(x)
        output = {
            "hidden_state": hidden_state,
            "mu": self.fc_mu(hidden_state),
        }
        if include_logvar:
            output["logvar"] = self.fc_logvar(hidden_state)
        return output


class NoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = WNCausalConv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        batch, _, frames = x.shape
        noise = torch.randn((batch, 1, frames), device=x.device, dtype=x.dtype)
        return x + noise * self.linear(x)


class CausalDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        groups=1,
        use_noise_block: bool = False,
    ):
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNCausalTransposeConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if use_noise_block:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                CausalResidualUnit(output_dim, dilation=1, groups=groups),
                CausalResidualUnit(output_dim, dilation=3, groups=groups),
                CausalResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        depthwise: bool = False,
        d_out: int = 1,
        use_noise_block: bool = False,
    ):
        super().__init__()
        if depthwise:
            layers = [
                WNCausalConv1d(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),
                WNCausalConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [
                WNCausalConv1d(input_channel, channels, kernel_size=7, padding=3)
            ]

        for idx, stride in enumerate(rates):
            input_dim = channels // 2**idx
            output_dim = channels // 2 ** (idx + 1)
            groups = output_dim if depthwise else 1
            layers += [
                CausalDecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    groups=groups,
                    use_noise_block=use_noise_block,
                )
            ]

        layers += [
            Snake1d(output_dim),
            WNCausalConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AudioVAE(nn.Module):

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_rate: float = 25.0,
        channels: int = 1,
        vae_dim: int = 64,
        encoder_dim: int = 128,
        encoder_rates: List[int] = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = None,
        depthwise: bool = True,
        use_noise_block: bool = False,
        weight_init_value: float = 2.0e-2,
        logvar_min: float = -30.0,
        logvar_max: float = 20.0,
        posterior_type: str = "standard",
        fix_std: float = 0.5,
        std_dist_type: str = "gaussian",
    ):
        super().__init__()
        encoder_rates = [3, 5, 8, 8] if encoder_rates is None else list(encoder_rates)
        decoder_rates = (
            list(reversed(encoder_rates)) if decoder_rates is None else list(decoder_rates)
        )
        posterior_type = str(posterior_type).lower()
        std_dist_type = str(std_dist_type).lower()
        if sample_rate not in {16000, 24000}:
            raise ValueError("AudioVAE expects sample_rate to be 16000 or 24000.")
        expected_hop_length = int(round(float(sample_rate) / float(frame_rate)))
        if expected_hop_length <= 0:
            raise ValueError("frame_rate must produce a positive hop length.")
        if abs(float(sample_rate) / float(expected_hop_length) - float(frame_rate)) > 1e-6:
            raise ValueError(
                f"sample_rate={sample_rate} is not evenly divisible by "
                f"frame_rate={frame_rate}."
            )
        if channels != 1:
            raise ValueError("AudioVAE expects channels=1.")
        if int(np.prod(encoder_rates)) != expected_hop_length:
            raise ValueError(
                "AudioVAE expects encoder_rates product to match "
                f"sample_rate/frame_rate={expected_hop_length}."
            )
        if int(np.prod(decoder_rates)) != expected_hop_length:
            raise ValueError(
                "AudioVAE expects decoder_rates product to match "
                f"sample_rate/frame_rate={expected_hop_length}."
            )
        if vae_dim != 64:
            raise ValueError("AudioVAE expects vae_dim=64.")
        if posterior_type not in {"standard", "sigma"}:
            raise ValueError(
                f"Unsupported posterior_type={posterior_type!r}. "
                "Expected 'standard' or 'sigma'."
            )
        if std_dist_type not in {"fix", "gaussian"}:
            raise ValueError(
                f"Unsupported std_dist_type={std_dist_type!r}. "
                "Expected 'fix' or 'gaussian'."
            )
        if fix_std <= 0:
            raise ValueError("fix_std must be positive.")

        self.config = {
            "sample_rate": sample_rate,
            "frame_rate": frame_rate,
            "channels": channels,
            "vae_dim": vae_dim,
            "encoder_dim": encoder_dim,
            "encoder_rates": list(encoder_rates),
            "decoder_dim": decoder_dim,
            "decoder_rates": list(decoder_rates),
            "depthwise": depthwise,
            "use_noise_block": use_noise_block,
            "weight_init_value": weight_init_value,
            "logvar_min": logvar_min,
            "logvar_max": logvar_max,
            "posterior_type": posterior_type,
            "fix_std": fix_std,
            "std_dist_type": std_dist_type,
        }
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.channels = channels
        self.vae_dim = vae_dim
        self.hop_length = expected_hop_length
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.posterior_type = posterior_type
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type

        self.encoder = CausalEncoder(
            d_model=encoder_dim,
            latent_dim=vae_dim,
            strides=encoder_rates,
            depthwise=depthwise,
        )
        self.decoder = CausalDecoder(
            input_channel=vae_dim,
            channels=decoder_dim,
            rates=decoder_rates,
            depthwise=depthwise,
            use_noise_block=use_noise_block,
        )
        self.apply(lambda module: init_weights(module, weight_init_value))

        if self.posterior_type == "sigma":
            self.encoder.fc_logvar.requires_grad_(False)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_config(self):
        return self.config

    def preprocess(self, audio_data, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected sample_rate={self.sample_rate}, got {sample_rate}")
        if audio_data.ndim != 3:
            raise ValueError(f"Expected audio shape [B, C, T], got {tuple(audio_data.shape)}")
        if audio_data.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channel(s), got {audio_data.shape[1]}")
        right_pad = (-audio_data.shape[-1]) % self.hop_length
        return F.pad(audio_data, (0, right_pad))

    def encode(self, audio_data):
        encoded = self.encoder(
            audio_data,
            include_logvar=self.posterior_type != "sigma",
        )
        mean = encoded["mu"].transpose(1, 2)
        if self.posterior_type == "sigma":
            if self.std_dist_type == "fix":
                logvar = torch.full_like(mean, math.log(self.fix_std**2))
            else:
                std = torch.randn(
                    mean.shape[0],
                    device=mean.device,
                    dtype=mean.dtype,
                ) * (self.fix_std / 0.8)
                while std.dim() < mean.dim():
                    std = std.unsqueeze(-1)
                logvar = torch.log(std.abs().clamp_min(1e-12).pow(2)).expand_as(mean)
            return VAEEncoderOutput(mean=mean, logvar=logvar, mean_kl=True)

        logvar = encoded["logvar"].clamp(self.logvar_min, self.logvar_max)
        return VAEEncoderOutput(mean=mean, logvar=logvar.transpose(1, 2))

    def decode(self, latents, channels_first: bool | None = None):
        if latents.ndim != 3:
            raise ValueError(f"Expected 3D latents, got shape {tuple(latents.shape)}")
        if channels_first is None:
            if latents.shape[-1] == self.vae_dim:
                channels_first = False
            elif latents.shape[1] == self.vae_dim:
                channels_first = True
            else:
                raise ValueError(
                    f"Cannot infer latent layout for shape {tuple(latents.shape)} "
                    f"and vae_dim={self.vae_dim}"
                )
        if not channels_first:
            latents = latents.transpose(1, 2)
        return self.decoder(latents).float()
