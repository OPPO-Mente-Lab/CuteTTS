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

"""Inference sequence containers and prefix assembly."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass
class CuteTTSSegment:
    input_ids: torch.LongTensor
    speech_tensor: torch.FloatTensor
    speech_mask: torch.BoolTensor
    speech_pad_mask: torch.BoolTensor
    speech_semantic_tensor: Optional[torch.FloatTensor] = None
    speaker_linear_mask: Optional[torch.BoolTensor] = None

    def to(self, *args, **kwargs) -> "CuteTTSSegment":
        values = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            values[name] = value.to(*args, **kwargs) if isinstance(value, torch.Tensor) else value
        return replace(self, **values)

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def total_length(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def audio_length(self) -> int:
        return int(self.speech_tensor.shape[1])


@dataclass
class SegmentManagerConfig:
    max_length: int = 4096
    batch_pad_token_id: int = 0
    audio_pad_token_id: int = 0
    speech_pad_value: float = 0.0
    audio_feat_dim: int = 64
    audio_patch_size: int = 1
    semantic_feat_dim: Optional[int] = None
    enable_semantic_features: bool = False
    padding_side: str = "left"

    def __post_init__(self) -> None:
        if self.max_length <= 0 or self.audio_feat_dim <= 0 or self.audio_patch_size <= 0:
            raise ValueError("Segment dimensions must be positive.")
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'.")


class SegmentManager:
    def __init__(self, config: SegmentManagerConfig):
        self.config = config

    def _feature_shape(self) -> tuple[int, ...]:
        if self.config.audio_patch_size == 1:
            return (self.config.audio_feat_dim,)
        return (self.config.audio_patch_size, self.config.audio_feat_dim)

    def _prepare_speech_tensor(self, speech: torch.Tensor) -> torch.Tensor:
        patch = int(self.config.audio_patch_size)
        feature = int(self.config.audio_feat_dim)
        if speech.dim() == 4:
            expected = (1, patch, feature)
            if (speech.shape[0], speech.shape[2], speech.shape[3]) != expected:
                raise ValueError(f"Unexpected patched speech shape: {tuple(speech.shape)}")
            return speech
        if speech.dim() != 3 or speech.shape[0] != 1 or speech.shape[2] != feature:
            raise ValueError(f"Unexpected speech shape: {tuple(speech.shape)}")
        if patch == 1:
            return speech
        padding = (-speech.shape[1]) % patch
        if padding:
            speech = torch.cat(
                [speech, speech.new_zeros((1, padding, feature))], dim=1
            )
        return speech.reshape(1, -1, patch, feature)

    def create_text_segment(self, input_ids, **_kwargs) -> CuteTTSSegment:
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long).reshape(1, -1)
        length = int(input_ids.shape[1])
        return CuteTTSSegment(
            input_ids=input_ids,
            speech_tensor=torch.zeros((1, 0, *self._feature_shape())),
            speech_mask=torch.zeros((1, length), dtype=torch.bool),
            speech_pad_mask=torch.zeros((1, 0), dtype=torch.bool),
        )

    def create_speaker_linear_segment(self, **_kwargs) -> CuteTTSSegment:
        return CuteTTSSegment(
            input_ids=torch.full((1, 1), self.config.batch_pad_token_id, dtype=torch.long),
            speech_tensor=torch.zeros((1, 0, *self._feature_shape())),
            speech_mask=torch.zeros((1, 1), dtype=torch.bool),
            speech_pad_mask=torch.zeros((1, 0), dtype=torch.bool),
            speaker_linear_mask=torch.ones((1, 1), dtype=torch.bool),
        )

    def create_speech_segment(
        self,
        speech_tensor: torch.Tensor,
        speech_semantic_tensor: torch.Tensor | None = None,
        **_kwargs,
    ) -> CuteTTSSegment:
        if speech_semantic_tensor is not None:
            raise ValueError("Semantic features are not part of the public models.")
        speech_tensor = self._prepare_speech_tensor(speech_tensor)
        length = int(speech_tensor.shape[1])
        return CuteTTSSegment(
            input_ids=torch.full((1, length), self.config.audio_pad_token_id, dtype=torch.long),
            speech_tensor=speech_tensor,
            speech_mask=torch.ones((1, length), dtype=torch.bool),
            speech_pad_mask=torch.ones((1, length), dtype=torch.bool),
        )

    def fuse_segments(self, segments: list[CuteTTSSegment]):
        if not segments:
            raise ValueError("At least one segment is required.")
        total = sum(segment.total_length for segment in segments)
        if total > self.config.max_length:
            raise ValueError(f"Prefix length {total} exceeds {self.config.max_length}.")
        audio_total = sum(segment.audio_length for segment in segments)
        input_ids = torch.full((1, total), self.config.audio_pad_token_id, dtype=torch.long)
        speech_mask = torch.zeros((1, total), dtype=torch.bool)
        speech = torch.full(
            (1, audio_total, *self._feature_shape()),
            self.config.speech_pad_value,
        )
        speech_pad_mask = torch.ones((1, audio_total), dtype=torch.bool)
        speaker_mask = (
            torch.zeros((1, total), dtype=torch.bool)
            if any(segment.speaker_linear_mask is not None for segment in segments)
            else None
        )
        boundaries = []
        sequence_position = 0
        audio_position = 0
        for index, segment in enumerate(segments):
            sequence_end = sequence_position + segment.total_length
            audio_end = audio_position + segment.audio_length
            input_ids[:, sequence_position:sequence_end] = segment.input_ids
            speech_mask[:, sequence_position:sequence_end] = segment.speech_mask
            speech[:, audio_position:audio_end] = segment.speech_tensor
            if speaker_mask is not None and segment.speaker_linear_mask is not None:
                speaker_mask[:, sequence_position:sequence_end] = segment.speaker_linear_mask
            boundaries.append(
                {
                    "segment_idx": index,
                    "seq_start": sequence_position,
                    "seq_end": sequence_end,
                    "audio_start": audio_position,
                    "audio_end": audio_end,
                }
            )
            sequence_position = sequence_end
            audio_position = audio_end
        return CuteTTSSegment(
            input_ids=input_ids,
            speech_tensor=speech,
            speech_mask=speech_mask,
            speech_pad_mask=speech_pad_mask,
            speaker_linear_mask=speaker_mask,
        ), boundaries
