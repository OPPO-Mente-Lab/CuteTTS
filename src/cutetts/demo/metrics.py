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

"""Server-side streaming generation metrics."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from threading import Lock

import torch


@dataclass(frozen=True)
class GenerationMetrics:
    ttfa_seconds: float
    generation_seconds: float
    audio_duration_seconds: float
    rtf: float
    patch_count: int
    total_samples: int
    sample_rate: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class MetricsRecorder:
    """Record PCM-ready timestamps before any WebSocket work is performed."""

    def __init__(self, t0: float, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        self.t0 = float(t0)
        self.sample_rate = int(sample_rate)
        self._first: float | None = None
        self._last: float | None = None
        self._chunks: list[torch.Tensor] = []
        self._lock = Lock()

    def add_chunk(
        self,
        chunk: torch.Tensor,
        *,
        timestamp: float | None = None,
    ) -> torch.Tensor:
        """Normalize one already-host-visible mono chunk and timestamp it."""

        host = chunk.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if host.ndim == 2 and host.size(0) == 1:
            host = host[0]
        elif host.ndim == 3 and host.size(0) == 1 and host.size(1) == 1:
            host = host[0, 0]
        if host.ndim != 1:
            raise ValueError(
                "PCM callback must return mono [S], [1,S], or [1,1,S], got "
                f"{tuple(host.shape)}."
            )
        if host.numel() <= 0:
            raise ValueError("PCM callback returned an empty chunk.")
        if not torch.isfinite(host).all():
            raise ValueError("PCM callback returned non-finite samples.")
        now = time.perf_counter() if timestamp is None else float(timestamp)
        with self._lock:
            if self._first is None:
                self._first = now
            self._last = now
            self._chunks.append(host)
        return host

    @property
    def patch_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    @property
    def total_samples(self) -> int:
        with self._lock:
            return sum(int(chunk.numel()) for chunk in self._chunks)

    @property
    def ttfa_seconds(self) -> float | None:
        with self._lock:
            return None if self._first is None else self._first - self.t0

    def waveform(self) -> torch.Tensor:
        with self._lock:
            if not self._chunks:
                return torch.empty(0, dtype=torch.float32)
            return torch.cat(list(self._chunks), dim=0)

    def finish(self) -> GenerationMetrics:
        with self._lock:
            if self._first is None or self._last is None or not self._chunks:
                raise RuntimeError("Inference completed without a PCM chunk.")
            samples = sum(int(chunk.numel()) for chunk in self._chunks)
            patches = len(self._chunks)
            first = self._first
            last = self._last
        duration = samples / self.sample_rate
        total = last - self.t0
        if duration <= 0.0 or total < 0.0:
            raise RuntimeError("Invalid streaming metric timestamps or sample count.")
        return GenerationMetrics(
            ttfa_seconds=first - self.t0,
            generation_seconds=total,
            audio_duration_seconds=duration,
            rtf=total / duration,
            patch_count=patches,
            total_samples=samples,
            sample_rate=self.sample_rate,
        )
