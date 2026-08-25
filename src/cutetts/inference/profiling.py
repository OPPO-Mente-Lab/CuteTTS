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

"""Low-overhead semantic stage timing for synchronous inference."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class InferenceStageProfiler:
    """Accumulate wall-clock time and invocation counts by semantic stage."""

    seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, name: str, elapsed_seconds: float) -> None:
        with self._lock:
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed_seconds
            self.counts[name] = self.counts.get(name, 0) + 1

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                name: {
                    "seconds": self.seconds[name],
                    "count": self.counts[name],
                }
                for name in sorted(self.seconds)
            }


def start_stage(profiler: InferenceStageProfiler | None) -> float | None:
    return None if profiler is None else time.perf_counter()


def finish_stage(
    profiler: InferenceStageProfiler | None,
    name: str,
    started_at: float | None,
) -> None:
    if profiler is not None and started_at is not None:
        profiler.add(name, time.perf_counter() - started_at)
