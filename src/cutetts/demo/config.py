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

"""Model discovery and runtime options for the local web demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from cutetts.runtime import resolve_device


MODEL_ID_BY_VARIANT = {"base": "CuteTTS", "distill": "CuteTTS-distill"}
REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer/tokenizer.model",
    "weights/tts/model.safetensors",
    "weights/audio_vae/config.json",
    "weights/audio_vae/model.safetensors",
    "weights/speaker_encoder/config.json",
    "weights/speaker_encoder/model.safetensors",
)


@dataclass(frozen=True)
class ModelOption:
    model_id: str
    label: str
    variant: str
    path: Path

    @property
    def defaults(self) -> dict[str, object]:
        if self.variant == "distill":
            return {
                "cfg_strength": 2.0,
                "diffusion_steps": 4,
                "diffusion_sway_coefficient": 0.0,
                "seed": 42,
                "max_decode_length": 750,
                "allowed_diffusion_steps": [1, 2, 4],
                "sway_supported": False,
            }
        return {
            "cfg_strength": 2.0,
            "diffusion_steps": 10,
            "diffusion_sway_coefficient": -0.8,
            "seed": 42,
            "max_decode_length": 750,
            "allowed_diffusion_steps": None,
            "sway_supported": True,
        }

    def public_dict(self) -> dict[str, object]:
        return {
            "id": self.model_id,
            "label": self.label,
            "variant": self.variant,
            "defaults": self.defaults,
        }


@dataclass(frozen=True)
class DeviceOption:
    device_id: str
    label: str
    resolved_device: str

    def public_dict(self) -> dict[str, str]:
        return {
            "id": self.device_id,
            "label": self.label,
            "resolved_device": self.resolved_device,
        }


@dataclass(frozen=True)
class DemoSettings:
    model_root: Path
    warmup_reference: Path
    initial_device: str = "auto"
    host: str = "127.0.0.1"
    port: int = 7860
    cpu_threads: int | None = None
    cpu_interop_threads: int = 1
    seed: int = 42
    reference_min_seconds: float = 2.0
    reference_max_seconds: float = 30.0
    max_upload_bytes: int = 50 * 1024 * 1024
    temporary_ttl_seconds: float = 60 * 60
    temporary_root: Path | None = None


def discover_models(model_root: str | Path) -> dict[str, ModelOption]:
    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")

    models: dict[str, ModelOption] = {}
    for candidate in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = candidate / "config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("model_type") != "cutetts":
            continue
        variant = str(config.get("variant", ""))
        model_id = MODEL_ID_BY_VARIANT.get(variant)
        if model_id is None:
            continue
        if any(not (candidate / relative).is_file() for relative in REQUIRED_MODEL_FILES):
            continue
        if model_id in models:
            raise ValueError(
                f"Multiple {model_id} model directories were found under {root}."
            )
        models[model_id] = ModelOption(
            model_id=model_id,
            label=model_id,
            variant=variant,
            path=candidate.resolve(),
        )

    if not models:
        expected = root / "CuteTTS"
        raise FileNotFoundError(
            f"No valid CuteTTS models were found under {root}. "
            f"Expected directories such as {expected}."
        )
    return models


def default_model_id(models: dict[str, ModelOption]) -> str:
    if "CuteTTS-distill" in models:
        return "CuteTTS-distill"
    return next(iter(models))


def available_devices() -> list[DeviceOption]:
    auto = str(resolve_device("auto"))
    devices = [DeviceOption("auto", f"Auto ({auto})", auto)]
    if torch.backends.mps.is_available():
        devices.append(DeviceOption("mps", "Apple MPS", "mps"))
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            device_id = f"cuda:{index}"
            try:
                name = torch.cuda.get_device_name(index)
            except Exception:
                name = f"CUDA device {index}"
            devices.append(DeviceOption(device_id, f"{device_id} · {name}", device_id))
    devices.append(DeviceOption("cpu", "CPU", "cpu"))
    return devices


def validate_device_choice(device: str, devices: list[DeviceOption]) -> str:
    value = str(device).strip().lower()
    if value == "cuda":
        value = "cuda:0"
    if value not in {option.device_id for option in devices}:
        choices = ", ".join(option.device_id for option in devices)
        raise ValueError(f"Unsupported device {device!r}; choose one of: {choices}.")
    return value
