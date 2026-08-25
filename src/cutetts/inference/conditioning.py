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

"""Inference guidance and prefix construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cutetts.modeling.segments import CuteTTSSegment


@dataclass(frozen=True)
class BranchPlan:
    include_reference: bool
    use_dit_speaker: bool
    lm_uncond: bool = False
    include_prompt: bool = False


@dataclass(frozen=True)
class GuidancePlan:
    conditional: BranchPlan
    unconditional: BranchPlan | None
    core_cfg_mode: str
    uncond_history_always_zero: bool = False

    @property
    def uses_lm_cfg(self) -> bool:
        return self.core_cfg_mode == "lm"


def build_guidance_plan(mode: str, cfg_mode: str, cfg_strength: float) -> GuidancePlan:
    if mode not in {"tts", "voice_clone"}:
        raise ValueError("Unsupported inference mode.")
    if cfg_mode not in {"nocfg", "lm"}:
        raise ValueError("Only LM CFG or distilled single-branch guidance is supported.")
    conditional = BranchPlan(
        include_reference=mode == "voice_clone",
        use_dit_speaker=mode == "voice_clone",
    )
    if cfg_mode == "nocfg" or cfg_strength == 0.0:
        return GuidancePlan(conditional, None, "nocfg")
    return GuidancePlan(
        conditional,
        BranchPlan(False, False, lm_uncond=True),
        "lm",
    )


def build_prefix_segment(
    processor,
    plan: BranchPlan,
    *,
    target_text: str,
    reference_features: torch.Tensor | None = None,
) -> CuteTTSSegment:
    manager = processor.segment_manager
    if plan.lm_uncond:
        segments = [manager.create_text_segment(processor.tokenizer.encode(processor.text_suffix_token))]
    elif plan.include_reference:
        if reference_features is None:
            raise ValueError("Reference features are required for voice cloning.")
        reference = manager.create_speech_segment(
            reference_features[None, ...]
        )
        segments = processor._reference_prompt_segments(target_text, reference)
    else:
        segments = [
            manager.create_text_segment(
                processor.tokenizer.encode(processor._text_only_prompt(target_text))
            )
        ]
    total = sum(segment.total_length for segment in segments)
    if total > manager.config.max_length:
        raise ValueError(
            f"Inference prefix length {total} exceeds {manager.config.max_length}."
        )
    return manager.fuse_segments(segments)[0]


def lm_speaker_for_branch(
    plan: BranchPlan,
    speaker_embedding: torch.Tensor | None,
) -> torch.Tensor | None:
    if not plan.include_reference:
        return None
    if speaker_embedding is None:
        raise ValueError("Voice cloning requires a speaker embedding.")
    return speaker_embedding


def dit_speaker_for_plan(
    plan: GuidancePlan,
    speaker_embedding: torch.Tensor | None,
) -> torch.Tensor | None:
    if speaker_embedding is None:
        return None
    conditional = speaker_embedding if plan.conditional.use_dit_speaker else torch.zeros_like(speaker_embedding)
    if plan.core_cfg_mode == "nocfg":
        return conditional
    assert plan.unconditional is not None
    unconditional = speaker_embedding if plan.unconditional.use_dit_speaker else torch.zeros_like(speaker_embedding)
    return torch.cat([conditional, unconditional], dim=0)


def initial_previous_from_prefix(
    _speech_features: torch.Tensor | None,
    _include_prompt: bool,
) -> None:
    return None
