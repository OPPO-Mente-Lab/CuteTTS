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

"""FastAPI routes and WebSocket protocol for the local CuteTTS demo."""

from __future__ import annotations

import asyncio
import contextlib
import math
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import DemoSettings
from .service import CuteTTSDemoService, DemoGenerationResult
from .storage import DemoStorage


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_TEXT_CHARACTERS = 2000
UPLOAD_CHUNK_BYTES = 1024 * 1024


def _validate_generation_request(
    payload: Any,
    defaults: dict[str, object],
) -> tuple[str, str, str | None, float, int, float | None, int]:
    if not isinstance(payload, dict):
        raise ValueError("The WebSocket request must be a JSON object.")
    mode = payload.get("mode")
    if mode not in {"tts", "reference"}:
        raise ValueError("mode must be 'tts' or 'reference'.")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")
    text = text.strip()
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError(f"text may contain at most {MAX_TEXT_CHARACTERS} characters.")
    reference_id = payload.get("reference_id")
    if mode == "reference":
        if not isinstance(reference_id, str) or not reference_id:
            raise ValueError("reference mode requires reference_id.")
    elif reference_id is not None and not isinstance(reference_id, str):
        raise ValueError("reference_id must be a string or null.")
    try:
        cfg_strength = float(payload.get("cfg_strength", defaults["cfg_strength"]))
        diffusion_steps = int(
            payload.get("diffusion_steps", defaults["diffusion_steps"])
        )
        seed = int(payload.get("seed", defaults["seed"]))
        sway_value = payload.get(
            "diffusion_sway_coefficient",
            defaults["diffusion_sway_coefficient"],
        )
        sway = None if sway_value is None else float(sway_value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid generation sampling parameters.") from error
    if not math.isfinite(cfg_strength) or cfg_strength < 0:
        raise ValueError("cfg_strength must be a finite non-negative number.")
    if defaults.get("allowed_diffusion_steps") is not None:
        allowed_steps = {int(value) for value in defaults["allowed_diffusion_steps"]}
        if diffusion_steps not in allowed_steps:
            choices = ", ".join(str(value) for value in sorted(allowed_steps))
            raise ValueError(f"diffusion_steps must be one of: {choices}.")
        if cfg_strength > 5:
            raise ValueError("CuteTTS-distill cfg_strength must be in [0, 5].")
        if sway not in {None, 0, 0.0}:
            raise ValueError("CuteTTS-distill does not expose sway sampling.")
        sway = 0.0
    else:
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive.")
        if sway is None or not math.isfinite(sway) or not -1 <= sway <= 2 / (math.pi - 2):
            raise ValueError("diffusion_sway_coefficient is outside its valid domain.")
    return mode, text, reference_id, cfg_strength, diffusion_steps, sway, seed


async def _periodic_cleanup(storage: DemoStorage) -> None:
    interval = min(300.0, max(10.0, storage.ttl_seconds / 2.0))
    while True:
        await asyncio.sleep(interval)
        storage.cleanup_expired()


def create_app(
    settings: DemoSettings,
    *,
    storage: DemoStorage | None = None,
    service: CuteTTSDemoService | None = None,
) -> FastAPI:
    """Create an app; dependency injection keeps API tests model-free."""

    storage = storage or DemoStorage(
        root=settings.temporary_root,
        ttl_seconds=settings.temporary_ttl_seconds,
    )
    service = service or CuteTTSDemoService(settings, storage)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        initialization = service.start_initialization(loop)
        cleanup = asyncio.create_task(_periodic_cleanup(storage))
        app.state.initialization = initialization
        try:
            yield
        finally:
            cleanup.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup
            service.shutdown()
            storage.close()

    app = FastAPI(
        title="CuteTTS Local Demo",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.demo_settings = settings
    app.state.demo_storage = storage
    app.state.demo_service = service
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return service.status()

    @app.post("/api/runtime/load", status_code=202)
    async def load_runtime(payload: dict[str, object] = Body(...)) -> dict[str, object]:
        model_id = payload.get("model")
        device = payload.get("device")
        if not isinstance(model_id, str) or not isinstance(device, str):
            raise HTTPException(
                status_code=422,
                detail="model and device must be strings.",
            )
        try:
            future = service.request_reload(
                asyncio.get_running_loop(),
                model_id,
                device,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if future is None:
            raise HTTPException(
                status_code=409,
                detail="The runtime is busy generating or loading a model.",
            )
        return {"accepted": True, "model": model_id, "device": device}

    @app.post("/api/references")
    async def upload_reference(file: UploadFile = File(...)) -> dict[str, object]:
        original_name = Path(file.filename or "reference.audio").name
        reference_id, destination = storage.allocate_reference_path()
        size = 0
        try:
            with destination.open("xb") as handle:
                while True:
                    chunk = await file.read(UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Reference audio exceeds the 50 MiB limit.",
                        )
                    handle.write(chunk)
            if size <= 0:
                raise HTTPException(status_code=422, detail="Reference audio is empty.")
            try:
                info = sf.info(destination)
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail="The upload is not an audio format supported by this server.",
                ) from exc
            if info.frames <= 0 or info.samplerate <= 0:
                raise HTTPException(status_code=422, detail="Reference audio contains no audio.")
            if info.channels not in {1, 2}:
                raise HTTPException(
                    status_code=422,
                    detail="Reference audio must be mono or stereo.",
                )
            duration = float(info.frames) / float(info.samplerate)
            if duration < settings.reference_min_seconds - 1e-9:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Reference audio must be at least "
                        f"{settings.reference_min_seconds:g} seconds."
                    ),
                )
            if duration > settings.reference_max_seconds + 1e-9:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Reference audio must be at most "
                        f"{settings.reference_max_seconds:g} seconds."
                    ),
                )
            record = storage.register_reference(
                reference_id=reference_id,
                path=destination,
                original_name=original_name,
                size_bytes=size,
                duration_seconds=duration,
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
            )
            return record.public_dict()
        except HTTPException:
            destination.unlink(missing_ok=True)
            raise
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=f"Invalid reference audio: {exc}") from exc
        finally:
            await file.close()

    @app.delete("/api/references/{reference_id}")
    async def delete_reference(reference_id: str) -> dict[str, bool]:
        try:
            deleted = storage.delete_reference(reference_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Reference was not found.")
        return {"deleted": True}

    @app.get("/api/results/{job_id}.wav")
    async def download_result(job_id: str) -> FileResponse:
        result = storage.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Generated WAV was not found.")
        return FileResponse(
            result.path,
            media_type="audio/wav",
            filename=f"cutetts-{job_id}.wav",
        )

    @app.websocket("/api/generate")
    async def generate(websocket: WebSocket) -> None:
        await websocket.accept()
        job_id: str | None = None
        reference_id: str | None = None
        reference_path: Path | None = None
        worker_started = False
        client_active = True
        try:
            try:
                payload = await websocket.receive_json()
                defaults = dict(service.status().get("defaults") or {})
                (
                    mode,
                    text,
                    requested_reference_id,
                    cfg_strength,
                    diffusion_steps,
                    diffusion_sway_coefficient,
                    seed,
                ) = _validate_generation_request(payload, defaults)
            except (ValueError, TypeError) as exc:
                await websocket.send_json(
                    {"type": "error", "code": "invalid_request", "message": str(exc)}
                )
                await websocket.close(code=1008)
                return

            if mode == "reference":
                assert requested_reference_id is not None
                reference = storage.acquire_reference(requested_reference_id)
                if reference is None:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "reference_not_found",
                            "message": "The uploaded reference no longer exists.",
                        }
                    )
                    await websocket.close(code=1008)
                    return
                reference_id = requested_reference_id
                reference_path = reference.path

            job_id = service.reserve_job()
            if job_id is None:
                state = service.status()
                code = "busy" if state.get("busy") else "not_ready"
                message = (
                    "Another generation is currently running."
                    if code == "busy"
                    else "The model is not ready."
                )
                await websocket.send_json(
                    {"type": "error", "code": code, "message": message}
                )
                await websocket.close(code=1013)
                return

            # The request is now accepted and owns the inference worker.  Start
            # the protocol clock before any response framing or request-level
            # model preprocessing.  Browser playback remains outside this
            # server-side measurement.
            loop = asyncio.get_running_loop()
            t0 = await loop.run_in_executor(service.executor, service.prepare_timing)
            state = service.status()
            await websocket.send_json(
                {
                    "type": "start",
                    "job_id": job_id,
                    "mode": mode,
                    "model": state.get("active_model"),
                    "device": state.get("resolved_device"),
                    "sample_rate": state.get("sample_rate", 24_000),
                    "channels": 1,
                    "pcm_format": "float32-le",
                }
            )

            pcm_queue: asyncio.Queue[torch.Tensor | dict[str, object] | None] = (
                asyncio.Queue()
            )

            def publish_chunk(chunk: torch.Tensor) -> None:
                if client_active:
                    loop.call_soon_threadsafe(pcm_queue.put_nowait, chunk)

            def publish_latency(ttfa_seconds: float) -> None:
                if client_active:
                    loop.call_soon_threadsafe(
                        pcm_queue.put_nowait,
                        {"type": "latency", "ttfa_seconds": ttfa_seconds},
                    )

            async def run_worker() -> DemoGenerationResult | Exception:
                try:
                    return await loop.run_in_executor(
                        service.executor,
                        lambda: service.generate(
                            job_id=job_id,
                            t0=t0,
                            mode=mode,
                            text=text,
                            reference=reference_path,
                            cfg_strength=cfg_strength,
                            diffusion_steps=diffusion_steps,
                            diffusion_sway_coefficient=diffusion_sway_coefficient,
                            seed=seed,
                            on_chunk=publish_chunk,
                            on_first_chunk=publish_latency,
                        ),
                    )
                except Exception as exc:
                    return exc
                finally:
                    service.release_job(job_id)
                    if reference_id is not None:
                        storage.release_reference(reference_id)
                    loop.call_soon_threadsafe(pcm_queue.put_nowait, None)

            worker = asyncio.create_task(run_worker())
            worker_started = True
            while True:
                item = await pcm_queue.get()
                if item is None:
                    break
                if isinstance(item, dict):
                    await websocket.send_json(item)
                else:
                    samples = np.asarray(item.numpy(), dtype="<f4")
                    await websocket.send_bytes(samples.tobytes(order="C"))

            outcome = await worker
            if isinstance(outcome, Exception):
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "generation_failed",
                        "message": f"{type(outcome).__name__}: {outcome}",
                    }
                )
                await websocket.close(code=1011)
                return
            await websocket.send_json(
                {
                    "type": "complete",
                    "job_id": outcome.job_id,
                    "metrics": outcome.metrics.to_dict(),
                    "patch_count": outcome.metrics.patch_count,
                    "download_url": outcome.download_url,
                }
            )
            await websocket.close(code=1000)
        except WebSocketDisconnect:
            client_active = False
            if worker_started:
                with contextlib.suppress(Exception):
                    await asyncio.shield(worker)
        finally:
            client_active = False
            if not worker_started:
                if job_id is not None:
                    service.release_job(job_id)
                if reference_id is not None:
                    storage.release_reference(reference_id)

    @app.post("/api/generate")
    async def generate_complete_wav(
        text: str = Form(...),
        mode: str = Form("tts"),
        cfg_strength: float = Form(2.0),
        diffusion_steps: int | None = Form(None),
        diffusion_sway_coefficient: float | None = Form(None),
        max_decode_length: int = Form(750),
        seed: int = Form(42),
        reference_audio: UploadFile | None = File(None),
    ) -> Response:
        if max_decode_length != 750:
            raise HTTPException(status_code=422, detail="max_decode_length must be 750.")
        mapped_mode = "reference" if mode in {"reference", "voice_clone"} else mode
        if mapped_mode not in {"tts", "reference"}:
            raise HTTPException(status_code=422, detail="Unsupported generation mode.")
        temporary_path: Path | None = None
        job_id: str | None = None
        try:
            if reference_audio is not None:
                suffix = Path(reference_audio.filename or "reference.wav").suffix or ".wav"
                payload = await reference_audio.read(settings.max_upload_bytes + 1)
                if len(payload) > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Reference exceeds 50 MiB.")
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                    handle.write(payload)
                    temporary_path = Path(handle.name)
                try:
                    info = sf.info(temporary_path)
                except Exception as error:
                    raise HTTPException(
                        status_code=422,
                        detail="reference_audio is not a supported audio file.",
                    ) from error
                duration = float(info.frames) / float(info.samplerate)
                if info.channels not in {1, 2}:
                    raise HTTPException(
                        status_code=422,
                        detail="reference_audio must be mono or stereo.",
                    )
                if not settings.reference_min_seconds <= duration <= settings.reference_max_seconds:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "reference_audio must be between "
                            f"{settings.reference_min_seconds:g} and "
                            f"{settings.reference_max_seconds:g} seconds."
                        ),
                    )
            if mapped_mode == "reference" and temporary_path is None:
                raise HTTPException(status_code=422, detail="Voice clone requires reference_audio.")
            job_id = service.reserve_job()
            if job_id is None:
                raise HTTPException(status_code=409, detail="The runtime is not ready.")
            loop = asyncio.get_running_loop()
            t0 = await loop.run_in_executor(service.executor, service.prepare_timing)
            outcome = await loop.run_in_executor(
                service.executor,
                lambda: service.generate(
                    job_id=job_id,
                    t0=t0,
                    mode=mapped_mode,
                    text=text,
                    reference=temporary_path,
                    cfg_strength=cfg_strength,
                    diffusion_steps=diffusion_steps,
                    diffusion_sway_coefficient=diffusion_sway_coefficient,
                    seed=seed,
                    on_chunk=lambda _chunk: None,
                    on_first_chunk=lambda _latency: None,
                ),
            )
            result = storage.get_result(outcome.job_id)
            if result is None:
                raise HTTPException(status_code=500, detail="Generated WAV was not saved.")
            return Response(result.path.read_bytes(), media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            if job_id is not None:
                service.release_job(job_id)
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if reference_audio is not None:
                await reference_audio.close()

    return app
