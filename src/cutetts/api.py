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

"""Public Python inference API."""

from __future__ import annotations

import math
import queue
import random
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cutetts.inference.conditioning import (
    build_guidance_plan,
    build_prefix_segment,
    dit_speaker_for_plan,
    initial_previous_from_prefix,
    lm_speaker_for_branch,
)
from cutetts.inference.generation import NaiveInferConfig, naive_ar_infer
from cutetts.modeling.sampling import set_sampler_compile_mode
from cutetts.runtime import RuntimeBundle, load_runtime, prepare_reference_audio


@dataclass(frozen=True)
class GenerationResult:
    waveform: torch.Tensor
    sample_rate: int


@dataclass(frozen=True)
class AudioChunk:
    """One decoded mono PCM chunk produced during streaming generation.

    ``waveform`` is a contiguous CPU float32 tensor with shape ``[1, samples]``.
    """

    waveform: torch.Tensor
    sample_rate: int


@dataclass(frozen=True)
class _StreamFailure:
    error: BaseException


class _StreamCancelled(Exception):
    pass


_STREAM_DONE = object()


class CuteTTS:
    """Load one CuteTTS model directory and synthesize individual utterances."""

    def __init__(self, runtime: RuntimeBundle):
        self.runtime = runtime

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "auto",
    ) -> "CuteTTS":
        runtime = load_runtime(model_dir, device)
        set_sampler_compile_mode(
            "eager" if runtime.model.device.type == "mps" else "full-sampler"
        )
        return cls(runtime)

    @property
    def variant(self) -> str:
        return self.runtime.variant

    @property
    def sample_rate(self) -> int:
        return self.runtime.sample_rate

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _prepare_branch(model, prefix, speaker_embedding):
        prefix = prefix.to(model.device)
        embeds, _contains_speech, speech_features = model.prepare_input_embeds(
            prefix,
            lm_speaker_embedding=speaker_embedding,
        )
        return embeds, speech_features

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        *,
        mode: str = "tts",
        reference_audio: str | Path | None = None,
        cfg_strength: float = 2.0,
        diffusion_steps: int | None = None,
        diffusion_sway_coefficient: float | None = None,
        max_decode_length: int = 750,
        seed: int = 42,
        show_progress: bool = True,
        pcm_chunk_callback: Callable[[torch.Tensor], None] | None = None,
    ) -> GenerationResult:
        text = str(text).strip()
        if not text:
            raise ValueError("text must not be empty.")
        if mode not in {"tts", "voice_clone"}:
            raise ValueError("mode must be 'tts' or 'voice_clone'.")
        if mode == "voice_clone" and reference_audio is None:
            raise ValueError("voice_clone requires reference_audio.")
        if not math.isfinite(cfg_strength) or cfg_strength < 0.0:
            raise ValueError("cfg_strength must be a finite non-negative number.")
        if max_decode_length <= 0:
            raise ValueError("max_decode_length must be positive.")

        if self.variant == "base":
            steps = 10 if diffusion_steps is None else int(diffusion_steps)
            sway = -0.8 if diffusion_sway_coefficient is None else float(
                diffusion_sway_coefficient
            )
            if steps <= 0:
                raise ValueError("diffusion_steps must be positive for CuteTTS.")
            if not -1.0 <= sway <= 2.0 / (math.pi - 2.0):
                raise ValueError("diffusion_sway_coefficient is outside its valid domain.")
            cfg_mode = "lm"
            ordinary_cfg = cfg_strength
            distilled_cfg = None
        else:
            steps = 4 if diffusion_steps is None else int(diffusion_steps)
            if steps not in {1, 2, 4}:
                raise ValueError("CuteTTS-distill supports diffusion_steps 1, 2, or 4.")
            if diffusion_sway_coefficient not in {None, 0, 0.0}:
                raise ValueError("CuteTTS-distill does not expose sway sampling.")
            if cfg_strength > 5.0:
                raise ValueError("CuteTTS-distill cfg_strength must be in [0, 5].")
            sway = 0.0
            cfg_mode = "nocfg"
            ordinary_cfg = 0.0
            distilled_cfg = cfg_strength

        self._seed(int(seed))
        processor = self.runtime.processor
        model = self.runtime.model
        plan = build_guidance_plan(mode, cfg_mode, ordinary_cfg)

        reference_features = None
        speaker_embedding = None
        if mode == "voice_clone":
            reference_wave, speaker_wave = prepare_reference_audio(
                reference_audio,
                self.runtime.sample_rate,
                int(self.runtime.speaker_encoder.sample_rate),
            )
            speaker_device = next(self.runtime.speaker_encoder.parameters()).device
            with torch.autocast(device_type=model.device.type, enabled=False):
                [[reference_features]] = processor.acoustic_batch_extractor(
                    [[reference_wave.to(processor.device)]],
                    processor.acoustic_feature_forward,
                )
                speaker_output = self.runtime.speaker_encoder(
                    speaker_wave.to(speaker_device),
                    int(self.runtime.speaker_encoder.sample_rate),
                )
            speaker_embedding = speaker_output["embedding"].float()

        cond_prefix = build_prefix_segment(
            processor,
            plan.conditional,
            target_text=text,
            reference_features=reference_features,
        )
        cond_embeds, cond_speech = self._prepare_branch(
            model,
            cond_prefix,
            lm_speaker_for_branch(plan.conditional, speaker_embedding),
        )

        uncond_embeds = None
        initial_uncond = None
        if plan.uses_lm_cfg:
            assert plan.unconditional is not None
            uncond_prefix = build_prefix_segment(
                processor,
                plan.unconditional,
                target_text=text,
                reference_features=reference_features,
            )
            uncond_embeds, uncond_speech = self._prepare_branch(
                model,
                uncond_prefix,
                lm_speaker_for_branch(plan.unconditional, speaker_embedding),
            )
            initial_uncond = initial_previous_from_prefix(
                uncond_speech,
                plan.unconditional.include_prompt,
            )

        infer_config = NaiveInferConfig(
            diffusion_steps=steps,
            cfg_strength=ordinary_cfg,
            cfg_mode=cfg_mode,
            max_decode_length=int(max_decode_length),
            diffusion_sway_coefficient=sway,
            distilled_cfg_strength=distilled_cfg,
        )
        result = naive_ar_infer(
            infer_config,
            processor,
            model,
            cond_embeds,
            uncond_embeds,
            speaker_embedding=dit_speaker_for_plan(plan, speaker_embedding),
            initial_previous_cond=initial_previous_from_prefix(
                cond_speech,
                plan.conditional.include_prompt,
            ),
            initial_uncond_previous_cond=initial_uncond,
            separate_cfg_previous_cond=plan.uses_lm_cfg,
            uncond_previous_cond_always_zero=plan.uncond_history_always_zero,
            use_tqdm=bool(show_progress),
            decode_each_patch=mode == "voice_clone" or pcm_chunk_callback is not None,
            pcm_chunk_callback=pcm_chunk_callback,
        )
        return GenerationResult(
            waveform=result.waveforms.detach().cpu(),
            sample_rate=self.runtime.sample_rate,
        )

    def generate_stream(
        self,
        text: str,
        *,
        mode: str = "tts",
        reference_audio: str | Path | None = None,
        cfg_strength: float = 2.0,
        diffusion_steps: int | None = None,
        diffusion_sway_coefficient: float | None = None,
        max_decode_length: int = 750,
        seed: int = 42,
        show_progress: bool = False,
    ) -> Iterator[AudioChunk]:
        """Yield decoded PCM chunks as autoregressive generation progresses.

        Generation runs in a worker thread so the caller can consume each chunk
        immediately. Closing the iterator stops generation at the next decoded
        patch. Use only one active generation per ``CuteTTS`` instance.
        """

        messages: queue.SimpleQueue[AudioChunk | _StreamFailure | object] = (
            queue.SimpleQueue()
        )
        cancelled = threading.Event()

        def emit(chunk: torch.Tensor) -> None:
            if cancelled.is_set():
                raise _StreamCancelled()
            waveform = chunk.detach().to(device="cpu", dtype=torch.float32)
            if waveform.ndim == 3 and waveform.size(0) == 1 and waveform.size(1) == 1:
                waveform = waveform.squeeze(1)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim != 2 or waveform.size(0) != 1:
                raise RuntimeError(
                    "Streaming decode must return mono audio with shape [1, samples], "
                    f"got {tuple(waveform.shape)}."
                )
            if waveform.size(1) == 0:
                raise RuntimeError("Streaming decode returned an empty audio chunk.")
            messages.put(
                AudioChunk(
                    waveform=waveform.contiguous(),
                    sample_rate=self.runtime.sample_rate,
                )
            )

        def run() -> None:
            try:
                self.generate(
                    text,
                    mode=mode,
                    reference_audio=reference_audio,
                    cfg_strength=cfg_strength,
                    diffusion_steps=diffusion_steps,
                    diffusion_sway_coefficient=diffusion_sway_coefficient,
                    max_decode_length=max_decode_length,
                    seed=seed,
                    show_progress=show_progress,
                    pcm_chunk_callback=emit,
                )
            except _StreamCancelled:
                pass
            except BaseException as error:
                messages.put(_StreamFailure(error))
            finally:
                messages.put(_STREAM_DONE)

        worker = threading.Thread(
            target=run,
            name="cutetts-python-stream",
            daemon=True,
        )
        worker.start()
        try:
            while True:
                message = messages.get()
                if message is _STREAM_DONE:
                    break
                if isinstance(message, _StreamFailure):
                    raise message.error
                assert isinstance(message, AudioChunk)
                yield message
        finally:
            cancelled.set()
            worker.join()
