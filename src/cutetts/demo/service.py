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

"""Model lifecycle and single-worker inference service for the web demo."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from loguru import logger

from cutetts import CuteTTS
from cutetts.runtime import resolve_device

from .config import (
    DemoSettings,
    DeviceOption,
    ModelOption,
    available_devices,
    default_model_id,
    discover_models,
    validate_device_choice,
)
from .metrics import GenerationMetrics, MetricsRecorder
from .storage import DemoStorage


SERVICE_PHASES = (
    "starting",
    "unloading",
    "loading_model",
    "warming_tts",
    "warming_reference",
    "ready",
    "busy",
    "error",
)
WARMUP_TEXT = "The model is warming up for local speech synthesis."


@dataclass(frozen=True)
class DemoGenerationResult:
    job_id: str
    metrics: GenerationMetrics
    download_url: str


class CuteTTSDemoService:
    """Own one resident runtime and serialize all PyTorch operations."""

    def __init__(self, settings: DemoSettings, storage: DemoStorage) -> None:
        self.settings = settings
        self.storage = storage
        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cutetts-inference",
        )
        self._initialization_future: asyncio.Future | None = None
        self._reload_future: asyncio.Future | None = None
        self._phase = "starting"
        self._error: str | None = None
        self._active_job_id: str | None = None
        self._model: CuteTTS | None = None
        self._active_model_id: str | None = None
        self._requested_device = settings.initial_device
        self._resolved_device: str | None = None
        self._threads_configured = False
        self._catalog_error: str | None = None
        try:
            self._models = discover_models(settings.model_root)
        except Exception as error:
            self._models = {}
            self._catalog_error = str(error)
        self._devices = available_devices()
        try:
            self._requested_device = validate_device_choice(
                settings.initial_device,
                self._devices,
            )
        except ValueError as error:
            self._catalog_error = str(error)
        self._target_model_id = (
            default_model_id(self._models) if self._models else None
        )

    def _set_phase(self, phase: str, *, error: str | None = None) -> None:
        if phase not in SERVICE_PHASES:
            raise ValueError(f"Unknown demo phase: {phase}")
        with self._state_lock:
            self._phase = phase
            self._error = error
        logger.info("CuteTTS demo state: {}", phase)

    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor

    @property
    def models(self) -> dict[str, ModelOption]:
        return self._models

    @property
    def devices(self) -> list[DeviceOption]:
        return self._devices

    def status(self) -> dict[str, object]:
        with self._state_lock:
            phase = self._phase
            error = self._error or self._catalog_error
            active_job_id = self._active_job_id
            active_model_id = self._active_model_id
            target_model_id = self._target_model_id
            requested_device = self._requested_device
            resolved_device = self._resolved_device
        selected = self._models.get(target_model_id or "")
        return {
            "phase": "error" if error and not self._models else phase,
            "ready": phase == "ready" and self._model is not None,
            "busy": phase == "busy",
            "loading": phase in {
                "starting",
                "unloading",
                "loading_model",
                "warming_tts",
                "warming_reference",
            },
            "error": error,
            "active_job_id": active_job_id,
            "models": [model.public_dict() for model in self._models.values()],
            "devices": [device.public_dict() for device in self._devices],
            "selected_model": target_model_id,
            "active_model": active_model_id,
            "requested_device": requested_device,
            "resolved_device": resolved_device,
            "device": resolved_device or requested_device,
            "variant": None if selected is None else selected.variant,
            "model": target_model_id,
            "model_label": None if selected is None else selected.label,
            "sample_rate": 24_000,
            "defaults": {} if selected is None else selected.defaults,
        }

    def start_initialization(self, loop: asyncio.AbstractEventLoop) -> asyncio.Future | None:
        if self._initialization_future is not None:
            return self._initialization_future
        if self._catalog_error or self._target_model_id is None:
            self._set_phase("error", error=self._catalog_error or "No model was found.")
            return None
        self._initialization_future = self._begin_reload(
            loop,
            self._target_model_id,
            self._requested_device,
        )
        return self._initialization_future

    def request_reload(
        self,
        loop: asyncio.AbstractEventLoop,
        model_id: str,
        device: str,
    ) -> asyncio.Future | None:
        if model_id not in self._models:
            raise ValueError(f"Unknown model {model_id!r}.")
        selected_device = validate_device_choice(device, self._devices)
        return self._begin_reload(loop, model_id, selected_device)

    def _begin_reload(
        self,
        loop: asyncio.AbstractEventLoop,
        model_id: str,
        device: str,
    ) -> asyncio.Future | None:
        if not self._operation_lock.acquire(blocking=False):
            return None
        with self._state_lock:
            self._target_model_id = model_id
            self._requested_device = device
            self._error = None
        self._set_phase("loading_model")
        future = loop.run_in_executor(
            self._executor,
            self._reload_worker,
            model_id,
            device,
        )
        self._reload_future = future
        return future

    def _configure_threads(self) -> None:
        if self._threads_configured:
            return
        cpu_threads = self.settings.cpu_threads
        if cpu_threads is None:
            cpu_threads = (
                4 if resolve_device(self._requested_device).type == "mps" else 16
            )
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(self.settings.cpu_interop_threads)
        except RuntimeError:
            pass
        torch.set_float32_matmul_precision("high")
        self._threads_configured = True

    @staticmethod
    def _synchronize(device: torch.device | None) -> None:
        if device is None:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def _unload_model(self) -> None:
        model = self._model
        if model is not None:
            self._synchronize(model.runtime.model.device)
        self._model = None
        self._active_model_id = None
        self._resolved_device = None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _warmup(self, model: CuteTTS, mode: str) -> None:
        chunks = 0

        def discard_chunk(chunk: torch.Tensor) -> None:
            nonlocal chunks
            if chunk.device.type != "cpu":
                raise RuntimeError("Warmup PCM callback must receive CPU audio.")
            chunks += 1

        kwargs: dict[str, object] = {
            "mode": mode,
            "seed": self.settings.seed,
            "max_decode_length": 1,
            "show_progress": False,
            "pcm_chunk_callback": discard_chunk,
        }
        if mode == "voice_clone":
            kwargs["reference_audio"] = self.settings.warmup_reference
        result = model.generate(WARMUP_TEXT, **kwargs)
        if chunks <= 0 or result.waveform.numel() <= 0:
            raise RuntimeError(f"{mode} warmup did not produce audio.")

    def _reload_worker(self, model_id: str, device: str) -> None:
        try:
            self._configure_threads()
            if not self.settings.warmup_reference.is_file():
                raise FileNotFoundError(
                    f"Warmup reference does not exist: {self.settings.warmup_reference}"
                )
            if self._model is not None:
                self._set_phase("unloading")
                self._unload_model()
            self._set_phase("loading_model")
            option = self._models[model_id]
            model = CuteTTS.from_pretrained(option.path, device=device)
            with self._state_lock:
                self._model = model
                self._active_model_id = model_id
                self._resolved_device = str(model.runtime.model.device)
            self._set_phase("warming_tts")
            self._warmup(model, "tts")
            self._set_phase("warming_reference")
            self._warmup(model, "voice_clone")
            self._synchronize(model.runtime.model.device)
            self._set_phase("ready")
        except Exception as error:
            logger.exception("CuteTTS demo runtime load failed")
            self._unload_model()
            self._set_phase("error", error=f"{type(error).__name__}: {error}")
        finally:
            self._operation_lock.release()

    def reserve_job(self) -> str | None:
        with self._state_lock:
            if self._phase != "ready" or self._model is None:
                return None
        if not self._operation_lock.acquire(blocking=False):
            return None
        job_id = uuid.uuid4().hex
        with self._state_lock:
            self._active_job_id = job_id
        self._set_phase("busy")
        return job_id

    def prepare_timing(self) -> float:
        model = self._model
        if model is None:
            raise RuntimeError("The model is not loaded.")
        self._synchronize(model.runtime.model.device)
        return time.perf_counter()

    def generate(
        self,
        *,
        job_id: str,
        t0: float,
        mode: str,
        text: str,
        reference: Path | None,
        cfg_strength: float,
        diffusion_steps: int | None,
        diffusion_sway_coefficient: float | None,
        seed: int,
        on_chunk: Callable[[torch.Tensor], None],
        on_first_chunk: Callable[[float], None],
    ) -> DemoGenerationResult:
        model = self._model
        if model is None or self._active_job_id != job_id:
            raise RuntimeError("Generation job is not reserved.")
        if mode not in {"tts", "reference"}:
            raise ValueError("mode must be 'tts' or 'reference'.")
        if mode == "reference" and reference is None:
            raise ValueError("Reference mode requires an uploaded audio file.")

        recorder = MetricsRecorder(t0, model.runtime.sample_rate)

        def record_and_publish(chunk: torch.Tensor) -> None:
            host = recorder.add_chunk(chunk)
            on_chunk(host)
            if recorder.patch_count == 1:
                latency = recorder.ttfa_seconds
                assert latency is not None
                on_first_chunk(latency)

        result = model.generate(
            text,
            mode="voice_clone" if mode == "reference" else "tts",
            reference_audio=reference,
            cfg_strength=cfg_strength,
            diffusion_steps=diffusion_steps,
            diffusion_sway_coefficient=diffusion_sway_coefficient,
            max_decode_length=750,
            seed=seed,
            show_progress=False,
            pcm_chunk_callback=record_and_publish,
        )
        metrics = recorder.finish()
        streamed = recorder.waveform()
        returned = result.waveform.detach().cpu().float().reshape(-1)
        if returned.numel() != streamed.numel() or not torch.equal(returned, streamed):
            raise RuntimeError("Final waveform does not match streamed PCM patches.")
        self.storage.save_result(job_id, streamed, result.sample_rate)
        logger.info(
            "CuteTTS generation complete: job={} model={} device={} "
            "ttfa={:.3f}s audio={:.3f}s generation={:.3f}s rtf={:.3f}",
            job_id,
            self._active_model_id,
            self._resolved_device,
            metrics.ttfa_seconds,
            metrics.audio_duration_seconds,
            metrics.generation_seconds,
            metrics.rtf,
        )
        return DemoGenerationResult(
            job_id=job_id,
            metrics=metrics,
            download_url=f"/api/results/{job_id}.wav",
        )

    def release_job(self, job_id: str) -> None:
        release_operation = False
        with self._state_lock:
            if self._active_job_id == job_id:
                self._active_job_id = None
                release_operation = True
                if self._model is not None:
                    self._phase = "ready"
        if release_operation and self._operation_lock.locked():
            self._operation_lock.release()

    def shutdown(self) -> None:
        if self._operation_lock.locked():
            future = self._reload_future or self._initialization_future
            if future is not None:
                try:
                    future.result()
                except Exception:
                    pass
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._unload_model()
