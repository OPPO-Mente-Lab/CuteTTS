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

"""Safe temporary reference uploads and generated WAV results."""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


@dataclass
class ReferenceRecord:
    reference_id: str
    path: Path
    original_name: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    created_at: float
    leases: int = 0

    def public_dict(self) -> dict[str, str | int | float]:
        value = asdict(self)
        value.pop("path")
        value.pop("created_at")
        value.pop("leases")
        return value


@dataclass
class ResultRecord:
    job_id: str
    path: Path
    created_at: float


class DemoStorage:
    """Own a private temp tree and expose records only by generated UUIDs."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        ttl_seconds: float = 3600.0,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        self.ttl_seconds = float(ttl_seconds)
        self._owned_root = root is None
        self.root = (
            Path(tempfile.mkdtemp(prefix="cutetts-demo-"))
            if root is None
            else Path(root).expanduser().resolve()
        )
        self.upload_dir = self.root / "references"
        self.result_dir = self.root / "results"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self._references: dict[str, ReferenceRecord] = {}
        self._results: dict[str, ResultRecord] = {}
        self._lock = threading.RLock()

    def allocate_reference_path(self) -> tuple[str, Path]:
        reference_id = uuid.uuid4().hex
        # SoundFile detects supported containers from their headers.  A neutral
        # server-side suffix avoids trusting or constraining the client name.
        return reference_id, self.upload_dir / f"{reference_id}.audio"

    def register_reference(
        self,
        *,
        reference_id: str,
        path: Path,
        original_name: str,
        size_bytes: int,
        duration_seconds: float,
        sample_rate: int,
        channels: int,
    ) -> ReferenceRecord:
        expected = (self.upload_dir / f"{reference_id}.audio").resolve()
        if path.resolve() != expected:
            raise ValueError("Reference path is outside the managed upload directory.")
        record = ReferenceRecord(
            reference_id=reference_id,
            path=expected,
            original_name=Path(original_name).name,
            size_bytes=int(size_bytes),
            duration_seconds=float(duration_seconds),
            sample_rate=int(sample_rate),
            channels=int(channels),
            created_at=time.time(),
        )
        with self._lock:
            self._references[reference_id] = record
        return record

    def get_reference(self, reference_id: str) -> ReferenceRecord | None:
        with self._lock:
            record = self._references.get(reference_id)
            if record is None or not record.path.is_file():
                return None
            return record

    def acquire_reference(self, reference_id: str) -> ReferenceRecord | None:
        with self._lock:
            record = self._references.get(reference_id)
            if record is None or not record.path.is_file():
                return None
            record.leases += 1
            return record

    def release_reference(self, reference_id: str) -> None:
        with self._lock:
            record = self._references.get(reference_id)
            if record is not None:
                record.leases = max(0, record.leases - 1)

    def delete_reference(self, reference_id: str) -> bool:
        with self._lock:
            record = self._references.get(reference_id)
            if record is None:
                return False
            if record.leases:
                raise RuntimeError("Reference is currently being used by a generation job.")
            self._references.pop(reference_id, None)
        record.path.unlink(missing_ok=True)
        return True

    def save_result(
        self,
        job_id: str,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> ResultRecord:
        if not uuid.UUID(job_id).hex == job_id.replace("-", ""):
            raise ValueError("job_id must be a UUID.")
        destination = (self.result_dir / f"{job_id}.wav").resolve()
        if destination.parent != self.result_dir.resolve():
            raise ValueError("Result path escaped the managed result directory.")
        samples = waveform.detach().cpu().float().reshape(-1).numpy()
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        sf.write(destination, pcm16, int(sample_rate), subtype="PCM_16", format="WAV")
        record = ResultRecord(job_id=job_id, path=destination, created_at=time.time())
        with self._lock:
            self._results[job_id] = record
        return record

    def get_result(self, job_id: str) -> ResultRecord | None:
        with self._lock:
            record = self._results.get(job_id)
            if record is None or not record.path.is_file():
                return None
            return record

    def cleanup_expired(self, *, now: float | None = None) -> None:
        cutoff = (time.time() if now is None else float(now)) - self.ttl_seconds
        with self._lock:
            expired_references = [
                key
                for key, value in self._references.items()
                if value.created_at < cutoff and value.leases == 0
            ]
            expired_results = [
                key
                for key, value in self._results.items()
                if value.created_at < cutoff
            ]
            reference_records = [self._references.pop(key) for key in expired_references]
            result_records = [self._results.pop(key) for key in expired_results]
        for record in reference_records:
            record.path.unlink(missing_ok=True)
        for record in result_records:
            record.path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._owned_root:
            shutil.rmtree(self.root, ignore_errors=True)
