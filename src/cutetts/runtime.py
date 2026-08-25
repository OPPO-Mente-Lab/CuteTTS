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

"""Self-contained model and audio loading for CuteTTS inference."""

from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

from cutetts.audio_codec.model.speaker_encoder import FbankECAPAStudent
from cutetts.modeling.configuration import CuteTTSConfig
from cutetts.modeling.model import CuteTTSModel
from cutetts.modeling.processor import CuteTTSProcessor
from cutetts.modeling.segments import SegmentManagerConfig


@dataclass(frozen=True)
class RuntimeBundle:
    variant: str
    sample_rate: int
    model: CuteTTSModel
    processor: CuteTTSProcessor
    speaker_encoder: FbankECAPAStudent


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _load_strict(module: torch.nn.Module, path: Path) -> None:
    missing, unexpected = module.load_state_dict(load_file(str(path)), strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Weights did not load strictly from {path}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to the best available backend for this machine."""

    if not isinstance(device, str) or device.lower() != "auto":
        return torch.device(device)

    is_apple_silicon = (
        platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"}
    )
    if is_apple_silicon:
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_runtime(
    model_dir: str | Path,
    device: str | torch.device = "auto",
) -> RuntimeBundle:
    root = Path(model_dir).expanduser().resolve()
    config = _read_json(root / "config.json")
    if config.get("model_type") != "cutetts":
        raise ValueError(f"Not a CuteTTS model directory: {root}")
    variant = str(config.get("variant"))
    if variant not in {"base", "distill"}:
        raise ValueError(f"Unknown model variant: {variant!r}")

    device = resolve_device(device)
    attention = "sdpa"
    if device.type == "cuda":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            pass
        else:
            attention = "flash_attention_2"

    model_config = dict(config["architecture"])
    model_config["attn_implementation"] = attention
    model_config["use_pretrained_lm"] = False
    model_config["lm_model_name"] = None
    model = CuteTTSModel(CuteTTSConfig(**model_config))
    _load_strict(model, root / "weights" / "tts" / "model.safetensors")

    segment_config = SegmentManagerConfig(**config["processor"]["segment"])
    processor = CuteTTSProcessor(
        acoustic_vae_path=str(root / "weights" / "audio_vae"),
        semantic_vae_path=None,
        tokenizer=str(root / "tokenizer"),
        segment_cfg=segment_config,
        speech_compress_rate=int(config["processor"]["speech_compress_rate"]),
        feature_extractor_internal_batchsize=None,
        feature_extractor_sort_batches=True,
        sequence_style="auto_reference",
        enable_condition_drop=False,
        text_suffix_token="<|endofprompt|>",
        acoustic_feature_no_sampling=True,
        packing_enabled=False,
        speaker_embedding_enabled=True,
        speaker_embedding_source="precomputed",
        speaker_embedding_dim=256,
        lm_speaker_linear_enabled=True,
        text_only_prompt_style="instruction",
        reset_previous_cond_at_target_start=True,
        self_reference_mode="combined",
    )

    speaker_folder = root / "weights" / "speaker_encoder"
    speaker_config = _read_json(speaker_folder / "config.json")
    if speaker_config.pop("component", None) != "speaker_encoder":
        raise ValueError(f"Invalid speaker encoder component: {speaker_folder}")
    speaker_encoder = FbankECAPAStudent(**speaker_config)
    _load_strict(speaker_encoder, speaker_folder / "model.safetensors")

    model = model.to(device).eval()
    auxiliary_device = torch.device("cpu") if device.type == "mps" else device
    processor = processor.to(auxiliary_device).eval()
    speaker_encoder = speaker_encoder.float().to(auxiliary_device).eval()
    for module in (model, processor, speaker_encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return RuntimeBundle(
        variant=variant,
        sample_rate=int(config.get("sample_rate", 24000)),
        model=model,
        processor=processor,
        speaker_encoder=speaker_encoder,
    )


def _repeat_past_two_seconds(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    threshold = 2 * int(sample_rate)
    if waveform.numel() == 0:
        raise ValueError("Reference audio is empty.")
    if waveform.shape[-1] < threshold:
        repeats = math.ceil((threshold + 1) / waveform.shape[-1])
        waveform = waveform.repeat(1, repeats)
    return waveform


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if int(source_rate) == int(target_rate):
        return waveform
    import torchaudio.functional as audio_functional

    return audio_functional.resample(waveform, int(source_rate), int(target_rate))


def prepare_reference_audio(
    path: str | Path,
    reference_rate: int,
    speaker_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode once, crop to 30/8 seconds, and extend clips shorter than 2 seconds."""

    path = Path(path).expanduser().resolve()
    with sf.SoundFile(path, mode="r") as audio_file:
        source_rate = int(audio_file.samplerate)
        frames = min(len(audio_file), int(math.ceil(30.0 * source_rate)))
        audio = audio_file.read(frames=frames, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32)).mean(
        dim=0, keepdim=True
    )

    reference = _resample(waveform, source_rate, reference_rate)
    reference = reference[..., : 30 * int(reference_rate)]
    reference = _repeat_past_two_seconds(reference, reference_rate)

    speaker_source = waveform[..., : 8 * source_rate]
    speaker = _resample(speaker_source, source_rate, speaker_rate)
    speaker = speaker[..., : 8 * int(speaker_rate)]
    speaker = _repeat_past_two_seconds(speaker, speaker_rate)
    return reference.contiguous().float(), speaker.contiguous().float()
