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

"""Text and reference-audio preparation for inference."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from cutetts.modeling.audio_adapter import AudioAcousticVAEAdapter
from cutetts.modeling.segments import CuteTTSSegment, SegmentManager, SegmentManagerConfig
from cutetts.modeling.tokenizer import CuteTTSSentencePieceTokenizer
from cutetts.modeling.utils import BatchExtractor


def _load_tokenizer(path: str):
    root = Path(path).expanduser()
    if (root / "tokenizer.model").is_file():
        return CuteTTSSentencePieceTokenizer.from_pretrained(root)
    return AutoTokenizer.from_pretrained(root, trust_remote_code=True)


class CuteTTSProcessor(torch.nn.Module):
    def __init__(
        self,
        acoustic_vae_path: str,
        tokenizer: str,
        segment_cfg: SegmentManagerConfig,
        speech_compress_rate: int = 1920,
        text_suffix_token: str = "<|endofprompt|>",
        text_only_prompt_style: str = "instruction",
        lm_speaker_linear_enabled: bool = True,
        **_kwargs,
    ):
        super().__init__()
        if text_only_prompt_style != "instruction":
            raise ValueError("Only the public instruction prompt is supported.")
        if not lm_speaker_linear_enabled:
            raise ValueError("Speaker projection must remain enabled.")
        self.acoustic_vae = AudioAcousticVAEAdapter(acoustic_vae_path)
        self.tokenizer = _load_tokenizer(tokenizer)
        self.segment_manager = SegmentManager(segment_cfg)
        self.text_suffix_token = text_suffix_token
        self.text_only_prompt_style = text_only_prompt_style
        self.lm_speaker_linear_enabled = True
        self.acoustic_feature_no_sampling = True
        self.acoustic_batch_extractor = BatchExtractor(
            input_length_dim=1,
            output_length_dim=0,
            length_scale=1 / int(speech_compress_rate),
            internal_batch_size=None,
            sort_batches=True,
        )

    @property
    def has_semantic_vae(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        return next(self.acoustic_vae.parameters()).device

    def acoustic_feature_forward(self, batched_waves: torch.Tensor) -> torch.Tensor:
        if batched_waves.dim() == 2:
            batched_waves = batched_waves.unsqueeze(1)
        return self.acoustic_vae.encode(batched_waves).mean

    def _text_only_prompt(self, text: str) -> str:
        return (
            "Transform the text into speech output.\n"
            f"text input:\n{text}\n{self.text_suffix_token}"
        )

    def _reference_prompt_segments(
        self,
        target_text: str,
        reference_speech_segment: CuteTTSSegment,
    ) -> list[CuteTTSSegment]:
        manager = self.segment_manager
        prefix = manager.create_text_segment(
            self.tokenizer.encode(
                "Transform the text into speech output, utilizing the distinct voice "
                "of the provided speech sample.\nvoice reference:\n<|im_start|>"
            )
        )
        suffix = manager.create_text_segment(
            self.tokenizer.encode(
                f"<|im_end|>\ntext input:\n{target_text}\n{self.text_suffix_token}"
            )
        )
        return [
            prefix,
            manager.create_speaker_linear_segment(),
            manager.create_text_segment(
                self.tokenizer.encode("<|im_end|>\n<|im_start|>")
            ),
            reference_speech_segment,
            suffix,
        ]
