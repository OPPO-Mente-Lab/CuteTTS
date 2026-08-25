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

"""CuteTTS Naive Inference Module.

This module provides a naive implementation of autoregressive inference for the
CuteTTS (Continuous Text-to-Speech) model. It supports various classifier-free
guidance (CFG) modes.

The main components include:
- NaiveInferConfig: Configuration parameters for inference
- NaiveInferState: State management for autoregressive inference
- NaiveInferResult: Result container for generated waveforms
- naive_ar_infer: Main inference function

Example:
    >>> config = NaiveInferConfig(cfg_mode='lm', cfg_strength=0.5)
    >>> result = naive_ar_infer(config, processor, lm, input_embeds, uncond_embeds)
"""

import math
import queue
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from cutetts.inference.profiling import (
    InferenceStageProfiler,
    finish_stage,
    start_stage,
)
from cutetts.modeling.model import CuteTTSModel
from cutetts.modeling.processor import CuteTTSProcessor
from loguru import logger
from tqdm import tqdm
from transformers.cache_utils import Cache, DynamicCache, StaticCache


@dataclass
class NaiveInferConfig:
    """Configuration for naive inference.

    Attributes:
        diffusion_steps: Number of diffusion sampling steps.
        cfg_strength: Extra classifier-free guidance strength; 0.0 is ordinary conditional generation.
        cfg_mode: `lm` for the base model or `nocfg` for the distilled model.
        max_decode_length: Maximum number of autoregressive steps.
    """
    diffusion_steps: int = 20
    cfg_strength: float = 0.5
    cfg_mode: str = 'head'
    max_decode_length: int = 750
    diffusion_sway_coefficient: float = 0.0
    distilled_cfg_strength: "float|None" = None
    batch_lm_cfg_decode: bool = False
    static_lm_cache: bool = False
    compile_lm_decode: bool = False

    def __post_init__(self):
        assert self.cfg_mode in {'nocfg', 'lm'}
        assert self.cfg_strength >= 0.0
        assert self.distilled_cfg_strength is None or self.distilled_cfg_strength >= 0.0
        assert -1.0 <= self.diffusion_sway_coefficient <= 2.0 / (math.pi - 2.0)
        if self.distilled_cfg_strength is not None:
            assert self.cfg_mode == "nocfg"
            assert self.cfg_strength == 0.0
        if self.batch_lm_cfg_decode:
            assert self.cfg_mode == 'lm'
            assert self.cfg_strength > 0.0
            assert not self.static_lm_cache
        if self.compile_lm_decode:
            assert not self.batch_lm_cfg_decode


def _new_static_lm_cache(
    lm: CuteTTSModel,
    max_cache_len: int,
    slot: str,
) -> StaticCache:
    """Get an exact-capacity, fixed-address Qwen cache workspace."""
    if max_cache_len <= 0:
        raise ValueError("Static LM cache capacity must be positive.")
    pool_attribute = "_cutetts_static_lm_cache_pool"
    pool = getattr(lm, pool_attribute, None)
    if pool is None:
        pool = {}
        object.__setattr__(lm, pool_attribute, pool)
    key = (slot, max_cache_len)
    cached = pool.get(key)
    if cached is not None:
        cached.reset()
        return cached
    try:
        dtype = next(lm.qwen_backbone.parameters()).dtype
    except StopIteration:
        dtype = torch.float32
    cached = StaticCache(
        config=lm.qwen_backbone.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=lm.device,
        dtype=dtype,
    )
    pool[key] = cached
    return cached


def _compiled_cpu_lm_decode_target(lm: CuteTTSModel):
    """Return one lazily compiled Qwen forward used only by decode steps."""
    if lm.device.type != "cpu":
        raise ValueError("LM decode compilation is currently CPU-only.")
    attribute = "_cutetts_compiled_cpu_lm_decode"
    target = getattr(lm, attribute, None)
    if target is None:
        target = torch.compile(
            lm.qwen_backbone.forward,
            dynamic=True,
            fullgraph=False,
        )
        # Avoid registering the optimized callable as a second nn.Module child.
        object.__setattr__(lm, attribute, target)
    return target


def _forward_lm_decode(
    config: NaiveInferConfig,
    lm: CuteTTSModel,
    **kwargs,
):
    if not config.compile_lm_decode:
        return lm.forward_lm(**kwargs)
    return lm.forward_lm(
        **kwargs,
        lm_model=_compiled_cpu_lm_decode_target(lm),
    )


def _cfg_enabled(config: NaiveInferConfig) -> bool:
    return config.cfg_mode in ['head', 'lm', 'lm_spk', 'reference', 'reference_spk'] and config.cfg_strength > 0.0


def _lm_cfg_enabled(config: NaiveInferConfig) -> bool:
    return config.cfg_mode in ['lm', 'lm_spk', 'reference', 'reference_spk'] and config.cfg_strength > 0.0


def _add_sway_sampling_input(
    head_input: dict,
    config: NaiveInferConfig,
) -> None:
    if config.diffusion_sway_coefficient == 0.0:
        return
    head_input["sway_sampling_coefficient"] = (
        config.diffusion_sway_coefficient
    )


def _add_distilled_cfg_strength_input(
    head_input: dict,
    config: NaiveInferConfig,
) -> None:
    if config.distilled_cfg_strength is not None:
        head_input["distilled_cfg_strength"] = config.distilled_cfg_strength


def _new_sampling_condition_cache(
    lm: CuteTTSModel,
) -> list | None:
    """Create a request-local cache for fixed step-distilled conditions."""
    if bool(getattr(lm.head, "step_size_embedding_enabled", False)):
        return []
    return None


def _add_sampling_condition_cache_input(
    head_input: dict,
    sampling_condition_cache: list | None,
) -> None:
    if sampling_condition_cache is not None:
        head_input["sampling_condition_cache"] = sampling_condition_cache


def _latent_history_size(lm: CuteTTSModel) -> int:
    return int(getattr(lm.config, "diff_latent_history_size", 1 if getattr(lm.config, "diff_latent_history", False) else 0))


def _init_latent_history(lm: CuteTTSModel, dtype: torch.dtype) -> "torch.Tensor|None":
    history_size = _latent_history_size(lm)
    if history_size <= 0:
        return None
    return torch.zeros(
        (1, lm.config.acoustic_latent_dim * history_size),
        device=lm.device,
        dtype=dtype,
    )


def _add_latent_history_input(
    head_input: dict,
    latent_history: "torch.Tensor|None",
    use_cfg: bool,
) -> None:
    if latent_history is None:
        return
    head_input["history"] = latent_history.repeat(2, 1) if use_cfg else latent_history


def _update_latent_history(
    latent_history: "torch.Tensor|None",
    pred_latent: torch.Tensor,
) -> "torch.Tensor|None":
    if latent_history is None:
        return None
    latent_dim = pred_latent.size(-1)
    if latent_history.size(-1) == latent_dim:
        return pred_latent.to(dtype=latent_history.dtype)
    return torch.cat(
        [pred_latent.to(dtype=latent_history.dtype), latent_history[:, :-latent_dim]],
        dim=-1,
    )


def _uses_audio_dit_previous_cond(lm: CuteTTSModel) -> bool:
    return (
        getattr(lm.config, "diff_head_kind", None) == "audio_dit"
        and bool(getattr(lm.config, "diff_dit_use_previous_cond", True))
    )


def _init_audio_dit_previous_cond(
    lm: CuteTTSModel,
    dtype: torch.dtype,
    initial_previous_cond: "torch.Tensor|None" = None,
) -> "torch.Tensor|None":
    if not _uses_audio_dit_previous_cond(lm):
        return None
    patch_size = int(getattr(lm.config, "diff_dit_patch_size", 1))
    if patch_size != int(getattr(lm.config, "locenc_patch_size", patch_size)):
        raise ValueError("diff_dit_patch_size and locenc_patch_size must match for audio_dit inference.")
    expected_shape = (
        (1, patch_size, lm.config.acoustic_latent_dim)
        if patch_size != 1
        else (1, lm.config.acoustic_latent_dim)
    )
    if initial_previous_cond is not None:
        if tuple(initial_previous_cond.shape) != expected_shape:
            raise ValueError(
                "initial_previous_cond shape mismatch: "
                f"expected {expected_shape}, got {tuple(initial_previous_cond.shape)}."
            )
        return initial_previous_cond.to(device=lm.device, dtype=dtype)
    if patch_size != 1:
        return torch.zeros(
            (1, patch_size, lm.config.acoustic_latent_dim),
            device=lm.device,
            dtype=dtype,
        )
    return torch.zeros(
        (1, lm.config.acoustic_latent_dim),
        device=lm.device,
        dtype=dtype,
    )


def _add_audio_dit_cond_input(
    head_input: dict,
    previous_cond: "torch.Tensor|None",
    use_cfg: bool,
    uncond_previous_cond: "torch.Tensor|None" = None,
    secondary_uncond_previous_cond: "torch.Tensor|None" = None,
) -> None:
    if previous_cond is None:
        return
    if use_cfg:
        if secondary_uncond_previous_cond is not None:
            if uncond_previous_cond is None:
                raise ValueError("Primary unconditional previous condition is required for multi-CFG.")
            if not (
                previous_cond.shape
                == uncond_previous_cond.shape
                == secondary_uncond_previous_cond.shape
            ):
                raise ValueError("All multi-CFG previous conditions must have the same shape.")
            previous_cond = torch.cat(
                [
                    previous_cond,
                    uncond_previous_cond.to(previous_cond),
                    secondary_uncond_previous_cond.to(previous_cond),
                ],
                dim=0,
            )
        elif uncond_previous_cond is None:
            repeat_shape = (2,) + (1,) * (previous_cond.dim() - 1)
            previous_cond = previous_cond.repeat(*repeat_shape)
        else:
            if previous_cond.shape != uncond_previous_cond.shape:
                raise ValueError(
                    "Conditional and unconditional previous conditions must have the same shape: "
                    f"{tuple(previous_cond.shape)} vs {tuple(uncond_previous_cond.shape)}."
                )
            previous_cond = torch.cat(
                [previous_cond, uncond_previous_cond.to(previous_cond)],
                dim=0,
            )
    head_input["cond"] = previous_cond


def _add_speaker_embedding_input(
    head_input: dict,
    speaker_embedding: "torch.Tensor|None",
) -> None:
    if speaker_embedding is not None:
        head_input["speaker_embedding"] = speaker_embedding


def _update_audio_dit_previous_cond(
    previous_cond: "torch.Tensor|None",
    pred_latent: torch.Tensor,
) -> "torch.Tensor|None":
    if previous_cond is None:
        return None
    return pred_latent.to(dtype=previous_cond.dtype)


def _stop_after_current_patch(lm: CuteTTSModel, hidden: torch.Tensor) -> bool:
    """Predict whether the patch conditioned on hidden is the final patch."""
    if not lm.config.two_class_stop_predictor:
        return False
    stop_logits = lm.stop_predictor(hidden)
    return bool(torch.argmax(stop_logits, dim=1).item() == 1)


def _latent_sequence_for_decode(pred_latent: torch.Tensor) -> torch.Tensor:
    if pred_latent.dim() == 2:
        return pred_latent[:, None, :]
    if pred_latent.dim() == 3:
        return pred_latent
    raise ValueError(f"Unexpected predicted latent shape for decode: {tuple(pred_latent.shape)}.")


def _latent_sequence_for_lm(pred_latent: torch.Tensor) -> torch.Tensor:
    if pred_latent.dim() == 2:
        return pred_latent[:, None, :]
    if pred_latent.dim() == 3:
        return pred_latent[:, None, :, :]
    raise ValueError(f"Unexpected predicted latent shape for LM input: {tuple(pred_latent.shape)}.")


def _require_offline_decode_compatible(lm: CuteTTSModel) -> None:
    if getattr(lm.config, "include_semantic_latent", False):
        raise NotImplementedError(
            "Offline VAE decoding does not support include_semantic_latent=True. "
            "That model variant needs per-step decoded audio for semantic feedback."
        )

@dataclass
class NaiveInferState:
    """State management for autoregressive inference state.

    Maintains LM caches and position information for autoregressive generation.

    Attributes:
        output_chunks/acoustic_cache/semantic_cache: Legacy fields kept for state compatibility.
        lm_kvc: Key-value cache for language model.
        uncond_lm_kvc: Key-value cache for unconditional language model (CFG).
        ar_position_ids: Position IDs for autoregressive generation.
        uncond_ar_position_ids: Position IDs for unconditional generation.
    """
    lm_kvc: Cache = field(default_factory=lambda: DynamicCache())
    uncond_lm_kvc: Cache = field(default_factory=lambda: DynamicCache())
    secondary_uncond_lm_kvc: Cache = field(default_factory=lambda: DynamicCache())

    ar_position_ids: torch.Tensor = field(default_factory=lambda: torch.LongTensor([[-1]]))
    uncond_ar_position_ids: torch.Tensor = field(default_factory=lambda: torch.LongTensor([[-1]]))
    secondary_uncond_ar_position_ids: torch.Tensor = field(default_factory=lambda: torch.LongTensor([[-1]]))

    def reset(self) -> None:
        """Reset all state components for new inference."""
        self.lm_kvc = type(self.lm_kvc)()
        self.uncond_lm_kvc = type(self.uncond_lm_kvc)()
        self.secondary_uncond_lm_kvc = type(self.secondary_uncond_lm_kvc)()
        self.ar_position_ids[:] = -1
        self.uncond_ar_position_ids[:] = -1
        self.secondary_uncond_ar_position_ids[:] = -1

@dataclass
class _BatchedLMDecodeState:
    cache: DynamicCache
    attention_mask: torch.Tensor


def _require_dynamic_cache(cache: Cache, name: str) -> DynamicCache:
    if not isinstance(cache, DynamicCache):
        raise TypeError(
            f"Batched LM CFG decode requires DynamicCache for {name}, got "
            f"{type(cache).__name__}."
        )
    return cache


def _left_pad_cache_tensor(tensor: torch.Tensor, length: int) -> torch.Tensor:
    pad_length = length - int(tensor.size(-2))
    if pad_length < 0:
        raise ValueError("Cannot pad a cache tensor to a shorter sequence length.")
    if pad_length == 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[-2] = pad_length
    return torch.cat([tensor.new_zeros(pad_shape), tensor], dim=-2)


def _merge_lm_cfg_caches(
    conditional_cache: Cache,
    unconditional_cache: Cache,
    conditional_mask: torch.Tensor | None = None,
    unconditional_mask: torch.Tensor | None = None,
) -> _BatchedLMDecodeState:
    conditional_cache = _require_dynamic_cache(
        conditional_cache,
        "conditional_cache",
    )
    unconditional_cache = _require_dynamic_cache(
        unconditional_cache,
        "unconditional_cache",
    )
    if len(conditional_cache) != len(unconditional_cache):
        raise ValueError("Conditional and unconditional LM caches have different layer counts.")
    conditional_length = conditional_cache.get_seq_length()
    unconditional_length = unconditional_cache.get_seq_length()
    merged_length = max(conditional_length, unconditional_length)
    device = conditional_cache.key_cache[0].device
    if conditional_mask is None:
        conditional_mask = torch.ones(
            (1, conditional_length), device=device, dtype=torch.long
        )
    if unconditional_mask is None:
        unconditional_mask = torch.ones(
            (1, unconditional_length), device=device, dtype=torch.long
        )
    conditional_mask = torch.nn.functional.pad(
        conditional_mask,
        (merged_length - conditional_mask.size(1), 0),
    )
    unconditional_mask = torch.nn.functional.pad(
        unconditional_mask,
        (merged_length - unconditional_mask.size(1), 0),
    )

    merged_cache = DynamicCache()
    for layer_index, ((conditional_key, conditional_value), (unconditional_key, unconditional_value)) in enumerate(
        zip(conditional_cache, unconditional_cache)
    ):
        merged_cache.update(
            torch.cat(
                [
                    _left_pad_cache_tensor(conditional_key, merged_length),
                    _left_pad_cache_tensor(unconditional_key, merged_length),
                ],
                dim=0,
            ),
            torch.cat(
                [
                    _left_pad_cache_tensor(conditional_value, merged_length),
                    _left_pad_cache_tensor(unconditional_value, merged_length),
                ],
                dim=0,
            ),
            layer_index,
        )
    return _BatchedLMDecodeState(
        cache=merged_cache,
        attention_mask=torch.cat(
            [conditional_mask, unconditional_mask],
            dim=0,
        ),
    )


def _batched_lm_cfg_decode_step(
    lm: CuteTTSModel,
    state: _BatchedLMDecodeState,
    input_embeds: torch.Tensor,
    conditional_position_ids: torch.Tensor,
    unconditional_position_ids: torch.Tensor,
):
    if input_embeds.size(0) != 1 or input_embeds.size(1) != 1:
        raise ValueError(
            "Batched LM CFG audio decode expects input_embeds with shape [1,1,D]."
        )
    state.attention_mask = torch.cat(
        [
            state.attention_mask,
            torch.ones(
                (2, 1),
                device=state.attention_mask.device,
                dtype=state.attention_mask.dtype,
            ),
        ],
        dim=1,
    )
    return lm.forward_lm(
        inputs_embeds=input_embeds.expand(2, -1, -1),
        attention_mask=state.attention_mask,
        position_ids=torch.cat(
            [conditional_position_ids, unconditional_position_ids],
            dim=0,
        ),
        past_key_values=state.cache,
        use_cache=True,
        output_attentions=False,
    )


def _conditional_lm_text_step_with_batched_cache(
    lm: CuteTTSModel,
    state: _BatchedLMDecodeState,
    input_embeds: torch.Tensor,
    position_ids: torch.Tensor,
):
    """Advance only the conditional branch, preserving the batched CFG cache."""
    conditional_cache, unconditional_cache = state.cache.batch_split(2, 1)
    conditional_mask = torch.cat(
        [
            state.attention_mask[:1],
            torch.ones(
                (1, input_embeds.size(1)),
                device=state.attention_mask.device,
                dtype=state.attention_mask.dtype,
            ),
        ],
        dim=1,
    )
    conditional_outputs = lm.forward_lm(
        inputs_embeds=input_embeds,
        attention_mask=conditional_mask,
        position_ids=position_ids,
        past_key_values=conditional_cache,
        use_cache=True,
        output_attentions=False,
    )
    merged = _merge_lm_cfg_caches(
        conditional_cache,
        unconditional_cache,
        conditional_mask,
        state.attention_mask[1:2],
    )
    state.cache = merged.cache
    state.attention_mask = merged.attention_mask
    return conditional_outputs

@dataclass
class NaiveInferResult:
    """Result container for naive inference.

    Attributes:
        waveforms: Generated audio waveforms with shape (batch_size, samples).
    """
    waveforms: torch.Tensor


def _vae_decode_autocast(processor: CuteTTSProcessor, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    try:
        vae_dtype = next(processor.acoustic_vae.parameters()).dtype
    except StopIteration:
        return nullcontext()
    if vae_dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type=device.type, dtype=vae_dtype)
    return torch.autocast(device_type=device.type, enabled=False)


def _offline_decode_generated_latents(
    processor: CuteTTSProcessor,
    lm: CuteTTSModel,
    latent_chunks: list[torch.Tensor],
) -> torch.Tensor:
    if not latent_chunks:
        return torch.empty((1, 0), device=lm.device)
    latent_sequence = torch.cat(
        [_latent_sequence_for_decode(chunk) for chunk in latent_chunks],
        dim=1,
    ).to(processor.device)
    with _vae_decode_autocast(processor, processor.device):
        waveforms = processor.acoustic_vae.decode(
            latent_sequence,
            use_cache=False,
            debug=False,
        )
    if waveforms.dim() == 3 and waveforms.size(1) == 1:
        return waveforms.squeeze(1)
    return waveforms


def _normalize_decoded_waveforms(waveforms: torch.Tensor) -> torch.Tensor:
    if waveforms.dim() == 3 and waveforms.size(1) == 1:
        return waveforms.squeeze(1)
    if waveforms.dim() != 2:
        raise ValueError(
            "Expected decoded waveforms with shape [B,S] or [B,1,S], got "
            f"{tuple(waveforms.shape)}."
        )
    return waveforms


def _decode_streaming_patch(
    processor: CuteTTSProcessor,
    lm: CuteTTSModel,
    vae_decoder,
    latent_patch: torch.Tensor,
) -> torch.Tensor | None:
    latent_sequence = _latent_sequence_for_decode(latent_patch).to(processor.device)
    with _vae_decode_autocast(processor, processor.device):
        waveforms = vae_decoder.decode_chunk(latent_sequence)
    if waveforms is None:
        return None
    return (
        _normalize_decoded_waveforms(waveforms)
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
    )


_PIPELINE_STOP = object()


class _MPSCPUStreamingDecodePipeline:
    """Overlap MPS latent generation with ordered CPU streaming VAE decode."""

    def __init__(
        self,
        processor: CuteTTSProcessor,
        pcm_chunk_callback: Callable[[torch.Tensor], None] | None,
        *,
        max_pending_patches: int = 2,
    ) -> None:
        if max_pending_patches <= 0:
            raise ValueError("max_pending_patches must be positive.")
        self._processor = processor
        self._pcm_chunk_callback = pcm_chunk_callback
        self._queue: queue.Queue[torch.Tensor | object] = queue.Queue(
            maxsize=max_pending_patches
        )
        self._chunks: list[torch.Tensor] = []
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="cutetts-cpu-vae-decode",
            daemon=True,
        )

    def __enter__(self) -> "_MPSCPUStreamingDecodePipeline":
        self._thread.start()
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        try:
            self.finish()
        except BaseException:
            if exc_type is None:
                raise

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _put(self, item: torch.Tensor | object) -> None:
        while True:
            self._raise_worker_error()
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def decode_chunk(self, latent_chunk: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("The streaming decode pipeline is already closed.")
        if latent_chunk.device.type != "cpu":
            raise ValueError("The CPU VAE pipeline requires a CPU latent patch.")
        self._put(latent_chunk.detach().contiguous())

    def _worker(self) -> None:
        try:
            streaming_decode = getattr(
                self._processor.acoustic_vae,
                "streaming_decode",
                None,
            )
            if not callable(streaming_decode):
                raise TypeError(
                    "The acoustic VAE does not provide streaming_decode()."
                )
            with streaming_decode() as decoder:
                while True:
                    latent_chunk = self._queue.get()
                    if latent_chunk is _PIPELINE_STOP:
                        break
                    assert isinstance(latent_chunk, torch.Tensor)
                    with _vae_decode_autocast(
                        self._processor,
                        self._processor.device,
                    ):
                        waveform = decoder.decode_chunk(latent_chunk)
                    chunk = (
                        _normalize_decoded_waveforms(waveform)
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                    )
                    self._chunks.append(chunk)
                    if self._pcm_chunk_callback is not None:
                        self._pcm_chunk_callback(chunk)
        except BaseException as error:
            self._error = error

    def finish(self) -> list[torch.Tensor]:
        if not self._closed:
            try:
                self._put(_PIPELINE_STOP)
            except BaseException:
                pass
            self._thread.join()
            self._closed = True
        self._raise_worker_error()
        return list(self._chunks)


def _concat_streaming_waveform_chunks(
    chunks: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if not chunks:
        return torch.empty((1, 0), device=device)
    batch_size = chunks[0].size(0)
    if any(chunk.dim() != 2 or chunk.size(0) != batch_size for chunk in chunks):
        raise ValueError("Streaming VAE returned incompatible waveform chunk shapes.")
    return torch.cat(chunks, dim=-1)


def _module_dtype(module: torch.nn.Module, default: torch.dtype) -> torch.dtype:
    param = next(module.parameters(), None)
    return param.dtype if param is not None else default


def _head_condition_dtype(lm: CuteTTSModel) -> torch.dtype:
    return _module_dtype(lm.head, torch.float32)


def _acoustic_connector_dtype(lm: CuteTTSModel) -> torch.dtype:
    if hasattr(lm, "acoustic_embedding_dtype"):
        return lm.acoustic_embedding_dtype()
    return _module_dtype(lm.acoustic_connector, torch.float32)


@torch.no_grad()
def _naive_ar_infer_impl(
    config: NaiveInferConfig,
    processor: CuteTTSProcessor,
    lm: CuteTTSModel,
    prefix_embeds: torch.Tensor,
    uncond_prefix_embeds: "None|torch.Tensor",
    secondary_uncond_prefix_embeds: "None|torch.Tensor" = None,
    speaker_embedding: "None|torch.Tensor" = None,
    initial_previous_cond: "None|torch.Tensor" = None,
    initial_uncond_previous_cond: "None|torch.Tensor" = None,
    initial_secondary_uncond_previous_cond: "None|torch.Tensor" = None,
    separate_cfg_previous_cond: bool = False,
    uncond_previous_cond_always_zero: bool = False,
    state: "None|NaiveInferState" = None,
    use_tqdm: bool = True,
    vae_decoder=None,
    pcm_chunk_callback: Callable[[torch.Tensor], None] | None = None,
    stage_profiler: InferenceStageProfiler | None = None,
) -> NaiveInferResult:
    """Perform naive autoregressive inference for CuteTTS.

    This function implements a straightforward autoregressive generation process
    that iteratively predicts acoustic latents and feeds latent embeddings back to
    the language model. It either collects the complete latent sequence for an
    offline decode or sends each patch to an active stateful VAE decoder.

    Args:
        config: Inference configuration parameters.
        processor: CuteTTS processor with VAE components.
        lm: CuteTTS language model.
        prefix_embeds: Input embeddings for prefill, shape (1, T_prefix, d_model).
        uncond_prefix_embeds: Unconditional embeddings for CFG, required when cfg_mode='lm' and cfg_strength > 0.
        state: Optional inference state for streaming.
        use_tqdm: Whether to show progress bar.

    Returns:
        NaiveInferResult containing generated waveforms.

    Note:
        - Only supports batch size 1.
        - Supports CFG modes: 'nocfg', 'head', 'lm', 'lm_spk',
          'reference', and 'reference_spk'.
        - Uses offline decode unless ``vae_decoder`` is provided.
    """

    if _lm_cfg_enabled(config):
        assert uncond_prefix_embeds is not None
    use_multi_cfg = False
    if use_multi_cfg:
        if not _lm_cfg_enabled(config):
            raise ValueError("Multi-CFG requires LM CFG as the primary branch.")
        if secondary_uncond_prefix_embeds is None:
            raise ValueError("Multi-CFG requires secondary_uncond_prefix_embeds.")

    bcsz, Tprefix, _d_model = prefix_embeds.shape
    assert bcsz == 1
    state = state or NaiveInferState()
    _require_offline_decode_compatible(lm)

    if config.static_lm_cache:
        state.lm_kvc = _new_static_lm_cache(
            lm,
            Tprefix + config.max_decode_length,
            "naive_conditional",
        )
        if _lm_cfg_enabled(config):
            assert uncond_prefix_embeds is not None
            state.uncond_lm_kvc = _new_static_lm_cache(
                lm,
                int(uncond_prefix_embeds.size(1)) + config.max_decode_length,
                "naive_unconditional",
            )
        if use_multi_cfg:
            assert secondary_uncond_prefix_embeds is not None
            state.secondary_uncond_lm_kvc = _new_static_lm_cache(
                lm,
                int(secondary_uncond_prefix_embeds.size(1))
                + config.max_decode_length,
                "naive_secondary_unconditional",
            )


    prefill_started = start_stage(stage_profiler)
    state.ar_position_ids[:] = Tprefix
    state.ar_position_ids = state.ar_position_ids.to(lm.device)
    lm_outputs = lm.forward_lm(
        inputs_embeds=prefix_embeds.to(lm.device),
        attention_mask=None,
        position_ids=torch.arange(Tprefix, device=lm.device).reshape(1, -1),
        past_key_values=state.lm_kvc,
        use_cache=True,
        output_attentions=False
    )

    if _lm_cfg_enabled(config):
        _, Tprefix_uncond, _ = uncond_prefix_embeds.shape
        state.uncond_ar_position_ids[:] = Tprefix_uncond
        state.uncond_ar_position_ids = state.uncond_ar_position_ids.to(lm.device)
        lm_outputs_uncond = lm.forward_lm(
            inputs_embeds=uncond_prefix_embeds.to(lm.device),
            attention_mask=None,
            position_ids=torch.arange(Tprefix_uncond, device=lm.device).reshape(1, -1),
            past_key_values=state.uncond_lm_kvc,
            use_cache=True,
            output_attentions=False
        )
    if use_multi_cfg:
        _, Tprefix_secondary, _ = secondary_uncond_prefix_embeds.shape
        state.secondary_uncond_ar_position_ids[:] = Tprefix_secondary
        state.secondary_uncond_ar_position_ids = state.secondary_uncond_ar_position_ids.to(lm.device)
        lm_outputs_secondary_uncond = lm.forward_lm(
            inputs_embeds=secondary_uncond_prefix_embeds.to(lm.device),
            attention_mask=None,
            position_ids=torch.arange(Tprefix_secondary, device=lm.device).reshape(1, -1),
            past_key_values=state.secondary_uncond_lm_kvc,
            use_cache=True,
            output_attentions=False,
        )
    finish_stage(stage_profiler, "lm_prefill", prefill_started)

    last_hidden = lm_outputs.last_hidden_state[:, -1, :]
    last_hidden_uncond = (
        lm_outputs_uncond.last_hidden_state[:, -1, :]
        if _lm_cfg_enabled(config)
        else None
    )
    last_hidden_secondary_uncond = (
        lm_outputs_secondary_uncond.last_hidden_state[:, -1, :]
        if use_multi_cfg
        else None
    )
    batched_lm_decode_state = None

    head_dtype = _head_condition_dtype(lm)
    acoustic_connector_dtype = _acoustic_connector_dtype(lm)

    latent_history = _init_latent_history(lm, head_dtype)
    audio_dit_previous_cond = _init_audio_dit_previous_cond(
        lm,
        head_dtype,
        initial_previous_cond=initial_previous_cond,
    )
    audio_dit_uncond_previous_cond = None
    if separate_cfg_previous_cond and _lm_cfg_enabled(config):
        audio_dit_uncond_previous_cond = _init_audio_dit_previous_cond(
            lm,
            head_dtype,
            initial_previous_cond=initial_uncond_previous_cond,
        )
    audio_dit_secondary_uncond_previous_cond = None
    if use_multi_cfg:
        audio_dit_secondary_uncond_previous_cond = _init_audio_dit_previous_cond(
            lm,
            head_dtype,
            initial_previous_cond=initial_secondary_uncond_previous_cond,
        )
    generated_latents: list[torch.Tensor] = []
    decoded_waveform_chunks: list[torch.Tensor] = []
    sampling_condition_cache = _new_sampling_condition_cache(lm)

    for idx_step in (tqdm if use_tqdm else lambda x: x)(range(config.max_decode_length)):

        last_hidden_for_head = last_hidden.to(dtype=head_dtype)

        stop_after_current_patch = _stop_after_current_patch(lm, last_hidden)

        use_cfg = _cfg_enabled(config)
        head_additional_input=dict(
            num_sampling_steps=config.diffusion_steps,
            cfg=config.cfg_strength if use_cfg else 0.0,
        )
        _add_sway_sampling_input(head_additional_input, config)
        _add_distilled_cfg_strength_input(head_additional_input, config)
        _add_sampling_condition_cache_input(
            head_additional_input,
            sampling_condition_cache,
        )
        _add_latent_history_input(head_additional_input, latent_history, use_cfg)
        _add_audio_dit_cond_input(
            head_additional_input,
            audio_dit_previous_cond,
            use_cfg,
            audio_dit_uncond_previous_cond,
            audio_dit_secondary_uncond_previous_cond,
        )
        _add_speaker_embedding_input(head_additional_input, speaker_embedding)

        diffusion_started = start_stage(stage_profiler)
        if use_cfg and config.cfg_mode == 'head':
            # Head-level CFG uses a zero vector for the unconditional branch.
            pred_latent = lm.head.sample(
                torch.cat([last_hidden_for_head, torch.zeros_like(last_hidden_for_head)], dim=0),
                **head_additional_input
            )
        elif not use_cfg:
            pred_latent = lm.head.sample(
                last_hidden_for_head,
                **head_additional_input
            )
        elif _lm_cfg_enabled(config):
            # LM-level CFG uses the hidden state from a zero-prefix branch.
            assert last_hidden_uncond is not None
            hidden_branches = [
                last_hidden_for_head,
                last_hidden_uncond.to(dtype=head_dtype),
            ]
            if use_multi_cfg:
                assert last_hidden_secondary_uncond is not None
                hidden_branches.append(
                    last_hidden_secondary_uncond.to(dtype=head_dtype)
                )
            pred_latent = lm.head.sample(torch.cat(hidden_branches, dim=0), **head_additional_input)
        else: raise NotImplementedError()
        finish_stage(stage_profiler, "diffusion", diffusion_started)
        # Convert the predicted latent back to the VAE feature scale.
        pred_latent_scaled = pred_latent / lm.speech_scaling_factor - lm.speech_bias_factor
        if vae_decoder is None:
            generated_latents.append(pred_latent_scaled)
        else:
            vae_decode_started = start_stage(stage_profiler)
            decoded_chunk = _decode_streaming_patch(
                processor,
                lm,
                vae_decoder,
                pred_latent_scaled,
            )
            if decoded_chunk is not None:
                decoded_waveform_chunks.append(decoded_chunk)
                if pcm_chunk_callback is not None:
                    pcm_chunk_callback(decoded_chunk)
            finish_stage(stage_profiler, "vae_decode", vae_decode_started)

        if stop_after_current_patch:
            break

        latent_history = _update_latent_history(latent_history, pred_latent)
        audio_dit_previous_cond = _update_audio_dit_previous_cond(audio_dit_previous_cond, pred_latent)
        if audio_dit_uncond_previous_cond is not None:
            if uncond_previous_cond_always_zero:
                audio_dit_uncond_previous_cond.zero_()
            else:
                audio_dit_uncond_previous_cond = _update_audio_dit_previous_cond(
                    audio_dit_uncond_previous_cond,
                    pred_latent,
                )
        if audio_dit_secondary_uncond_previous_cond is not None:
            audio_dit_secondary_uncond_previous_cond = _update_audio_dit_previous_cond(
                audio_dit_secondary_uncond_previous_cond,
                pred_latent,
            )

        acoustic_embed = lm.embed_acoustic_latents(_latent_sequence_for_lm(pred_latent).to(dtype=acoustic_connector_dtype))
        input_embeds = acoustic_embed

        lm_decode_started = start_stage(stage_profiler)
        if config.batch_lm_cfg_decode:
            if batched_lm_decode_state is None:
                batched_lm_decode_state = _merge_lm_cfg_caches(
                    state.lm_kvc,
                    state.uncond_lm_kvc,
                )
                state.lm_kvc = batched_lm_decode_state.cache
                state.uncond_lm_kvc = DynamicCache()
            batched_outputs = _batched_lm_cfg_decode_step(
                lm,
                batched_lm_decode_state,
                input_embeds,
                state.ar_position_ids,
                state.uncond_ar_position_ids,
            )
            last_hidden = batched_outputs.last_hidden_state[:1, -1, :]
            last_hidden_uncond = batched_outputs.last_hidden_state[1:, -1, :]
            state.ar_position_ids += 1
            state.uncond_ar_position_ids += 1
        else:
            lm_outputs = _forward_lm_decode(
                config,
                lm,
                inputs_embeds=input_embeds,
                attention_mask=None,
                position_ids=state.ar_position_ids,
                past_key_values=state.lm_kvc,
                use_cache=True,
                output_attentions=False
            )
            last_hidden = lm_outputs.last_hidden_state[:, -1, :]
            state.ar_position_ids += 1

            if _lm_cfg_enabled(config):
                lm_outputs_uncond = _forward_lm_decode(
                    config,
                    lm,
                    inputs_embeds=input_embeds,
                    attention_mask=None,
                    position_ids=state.uncond_ar_position_ids,
                    past_key_values=state.uncond_lm_kvc,
                    use_cache=True,
                    output_attentions=False
                )
                last_hidden_uncond = lm_outputs_uncond.last_hidden_state[:, -1, :]
                state.uncond_ar_position_ids += 1
        if use_multi_cfg:
            lm_outputs_secondary_uncond = _forward_lm_decode(
                config,
                lm,
                inputs_embeds=input_embeds,
                attention_mask=None,
                position_ids=state.secondary_uncond_ar_position_ids,
                past_key_values=state.secondary_uncond_lm_kvc,
                use_cache=True,
                output_attentions=False,
            )
            last_hidden_secondary_uncond = (
                lm_outputs_secondary_uncond.last_hidden_state[:, -1, :]
            )
            state.secondary_uncond_ar_position_ids += 1
        finish_stage(stage_profiler, "lm_decode", lm_decode_started)

    if isinstance(vae_decoder, _MPSCPUStreamingDecodePipeline):
        decoded_waveform_chunks = vae_decoder.finish()

    final_decode_started = start_stage(stage_profiler)
    waveforms = (
        _offline_decode_generated_latents(processor, lm, generated_latents)
        if vae_decoder is None
        else _concat_streaming_waveform_chunks(decoded_waveform_chunks, processor.device)
    )
    finish_stage(stage_profiler, "final_waveform_assembly", final_decode_started)
    return NaiveInferResult(waveforms=waveforms)


@torch.no_grad()
def naive_ar_infer(
    config: NaiveInferConfig,
    processor: CuteTTSProcessor,
    lm: CuteTTSModel,
    prefix_embeds: torch.Tensor,
    uncond_prefix_embeds: "None|torch.Tensor",
    secondary_uncond_prefix_embeds: "None|torch.Tensor" = None,
    speaker_embedding: "None|torch.Tensor" = None,
    initial_previous_cond: "None|torch.Tensor" = None,
    initial_uncond_previous_cond: "None|torch.Tensor" = None,
    initial_secondary_uncond_previous_cond: "None|torch.Tensor" = None,
    separate_cfg_previous_cond: bool = False,
    uncond_previous_cond_always_zero: bool = False,
    state: "None|NaiveInferState" = None,
    use_tqdm: bool = True,
    decode_each_patch: bool = False,
    pcm_chunk_callback: Callable[[torch.Tensor], None] | None = None,
    stage_profiler: InferenceStageProfiler | None = None,
) -> NaiveInferResult:
    """Run ordinary AR inference with optional stateful per-patch VAE decode."""

    common_kwargs = dict(
        secondary_uncond_prefix_embeds=secondary_uncond_prefix_embeds,
        speaker_embedding=speaker_embedding,
        initial_previous_cond=initial_previous_cond,
        initial_uncond_previous_cond=initial_uncond_previous_cond,
        initial_secondary_uncond_previous_cond=initial_secondary_uncond_previous_cond,
        separate_cfg_previous_cond=separate_cfg_previous_cond,
        uncond_previous_cond_always_zero=uncond_previous_cond_always_zero,
        state=state,
        use_tqdm=use_tqdm,
        pcm_chunk_callback=pcm_chunk_callback,
        stage_profiler=stage_profiler,
    )
    if not decode_each_patch:
        return _naive_ar_infer_impl(
            config,
            processor,
            lm,
            prefix_embeds,
            uncond_prefix_embeds,
            **common_kwargs,
        )

    streaming_decode = getattr(processor.acoustic_vae, "streaming_decode", None)
    if not callable(streaming_decode):
        raise TypeError(
            "Per-patch reference decode requires an acoustic VAE with streaming_decode()."
        )
    if lm.device.type == "mps" and processor.device.type == "cpu":
        with _MPSCPUStreamingDecodePipeline(
            processor,
            pcm_chunk_callback,
        ) as vae_decoder:
            return _naive_ar_infer_impl(
                config,
                processor,
                lm,
                prefix_embeds,
                uncond_prefix_embeds,
                vae_decoder=vae_decoder,
                **{
                    **common_kwargs,
                    "pcm_chunk_callback": None,
                },
            )
    with streaming_decode() as vae_decoder:
        return _naive_ar_infer_impl(
            config,
            processor,
            lm,
            prefix_embeds,
            uncond_prefix_embeds,
            vae_decoder=vae_decoder,
            **common_kwargs,
        )
