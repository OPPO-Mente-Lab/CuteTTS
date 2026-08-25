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

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from cutetts.audio_codec.model.audio_vae import AudioVAE


@dataclass
class AudioEncodeOutput:
    mean: torch.Tensor
    std: None = None

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, *args, **kwargs) -> torch.Tensor:
        return self.mean


def is_audio_checkpoint(path: str | Path) -> bool:
    path = Path(path).expanduser()
    candidates = []
    if path.is_file():
        path = path.parent
    candidates.extend([path / "config.json"])
    for config_path in candidates:
        if not config_path.exists():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("component") == "audio_vae":
            return (config_path.parent / "model.safetensors").is_file()
    return False


class AudioAcousticVAEAdapter(nn.Module):
    """Expose the bundled acoustic autoencoder through the inference API."""

    def __init__(self, checkpoint_path: str | Path):
        super().__init__()
        folder = Path(checkpoint_path).expanduser().resolve()
        config_path = folder / "config.json"
        weights_path = folder / "model.safetensors"
        if not is_audio_checkpoint(folder):
            raise FileNotFoundError(f"Invalid audio VAE component: {folder}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.pop("component", None) != "audio_vae":
            raise ValueError(f"Invalid component declaration in {config_path}.")
        model = AudioVAE(
            **config,
        )
        missing, unexpected = model.load_state_dict(load_file(str(weights_path)), strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Audio VAE did not load strictly: missing={missing}, unexpected={unexpected}"
            )
        if int(getattr(model, "sample_rate", 0)) not in {16000, 24000}:
            raise ValueError(f"Unsupported sample rate: {getattr(model, 'sample_rate', None)}.")
        if int(getattr(model, "vae_dim", 0)) != 64:
            raise ValueError(f"Unsupported latent size: {getattr(model, 'vae_dim', None)}.")
        self.model = model.eval()
        self.sample_rate = int(model.sample_rate)
        self.channels = int(getattr(model, "channels", 1))
        self._streaming_decoder_active = False

    def encode(self, audio: torch.Tensor, *args, **kwargs) -> AudioEncodeOutput:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.dim() != 3:
            raise ValueError(f"Expected audio with shape [B, C, T] or [B, T], got {tuple(audio.shape)}.")
        padded = self.model.preprocess(audio, self.sample_rate)
        posterior = self.model.encode(padded)
        return AudioEncodeOutput(mean=posterior.mode())

    def decode(self, latent: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model.decode(latent).float()

    def streaming_decode(self) -> "AudioStreamingVAEDecoder":
        return AudioStreamingVAEDecoder(self)


class AudioStreamingVAEDecoder:
    """Decode new Audio latent patches while retaining causal-conv state."""

    def __init__(self, vae: AudioAcousticVAEAdapter):
        self._vae = vae
        self._states: dict[int, torch.Tensor] = {}
        self._originals: list[tuple[nn.Module, object]] = []

    def __enter__(self) -> "AudioStreamingVAEDecoder":
        if self._originals:
            raise RuntimeError("Streaming VAE decoder is already active.")
        if getattr(self._vae, "_streaming_decoder_active", False):
            raise RuntimeError("Another streaming VAE decoder is already active.")
        self._vae._streaming_decoder_active = True
        try:
            self._install()
        except Exception:
            self._restore()
            self._states.clear()
            self._vae._streaming_decoder_active = False
            raise
        return self

    def __exit__(self, *_exc) -> None:
        try:
            self._restore()
        finally:
            self._states.clear()
            self._vae._streaming_decoder_active = False

    def decode_chunk(self, latent_chunk: torch.Tensor) -> torch.Tensor:
        if not self._originals:
            raise RuntimeError("decode_chunk() must be called inside streaming_decode().")
        return self._vae.decode(latent_chunk)

    def _install(self) -> None:
        for module in self._vae.model.decoder.modules():
            causal_padding = getattr(module, "_CausalConv1d__padding", None)
            if isinstance(module, nn.Conv1d) and causal_padding is not None:
                output_padding = int(
                    getattr(module, "_CausalConv1d__output_padding", 0)
                )
                pad_size = int(causal_padding) * 2 - output_padding
                if pad_size > 0:
                    self._patch_causal_conv(module, pad_size)
                continue

            transpose_padding = getattr(
                module,
                "_CausalTransposeConv1d__padding",
                None,
            )
            if (
                isinstance(module, nn.ConvTranspose1d)
                and transpose_padding is not None
            ):
                output_padding = int(
                    getattr(module, "_CausalTransposeConv1d__output_padding", 0)
                )
                trim = int(transpose_padding) * 2 - output_padding
                context_size = (int(module.kernel_size[0]) - 1) // int(
                    module.stride[0]
                )
                if context_size > 0:
                    self._patch_transpose_conv(module, context_size, trim)

    def _patch_causal_conv(self, module: nn.Conv1d, pad_size: int) -> None:
        states = self._states
        key = id(module)
        original_forward = module.forward

        def forward(x, _key=key, _pad=pad_size, _module=module):
            if _key in states:
                padded = torch.cat([states[_key], x], dim=-1)
            else:
                padded = F.pad(x, (_pad, 0))

            if x.shape[-1] >= _pad:
                states[_key] = x[..., -_pad:].detach()
            else:
                previous = states.get(
                    _key,
                    x.new_zeros(x.shape[0], x.shape[1], _pad),
                )
                states[_key] = torch.cat([previous, x], dim=-1)[
                    ..., -_pad:
                ].detach()
            return nn.Conv1d.forward(_module, padded)

        module.forward = forward
        self._originals.append((module, original_forward))

    def _patch_transpose_conv(
        self,
        module: nn.ConvTranspose1d,
        context_size: int,
        trim: int,
    ) -> None:
        states = self._states
        key = id(module)
        original_forward = module.forward

        def forward(
            x,
            _key=key,
            _context=context_size,
            _trim=trim,
            _module=module,
        ):
            if _key in states:
                full_input = torch.cat([states[_key], x], dim=-1)
            else:
                full_input = F.pad(x, (_context, 0))
            states[_key] = full_input[..., -_context:].detach()

            output = nn.ConvTranspose1d.forward(_module, full_input)
            left = _context * int(_module.stride[0])
            if _trim > 0:
                return output[..., left:-_trim]
            return output[..., left:]

        module.forward = forward
        self._originals.append((module, original_forward))

    def _restore(self) -> None:
        for module, original_forward in reversed(self._originals):
            module.forward = original_forward
        self._originals.clear()
