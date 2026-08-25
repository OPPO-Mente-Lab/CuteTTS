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

"""Inference-only CuteTTS language and diffusion model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from cutetts.modeling.configuration import CuteTTSConfig
from cutetts.modeling.diffusion_head import AudioDiTHead, AudioLocEnc
from cutetts.modeling.segments import CuteTTSSegment


class CuteTTSModel(PreTrainedModel):
    config_class = CuteTTSConfig

    def __init__(self, config: CuteTTSConfig):
        super().__init__(config)
        if not config.locenc_enabled or config.include_semantic_latent:
            raise ValueError("The public models require the bundled acoustic encoder path.")
        if config.diff_head_kind != "audio_dit":
            raise ValueError("The public models require the bundled diffusion head.")
        if not config.lm_speaker_linear_enabled:
            raise ValueError("The public models require speaker conditioning.")

        self.locenc = AudioLocEnc(
            input_dim=config.acoustic_latent_dim,
            hidden_dim=config.locenc_hidden_dim,
            ffn_dim=config.locenc_ffn_dim,
            num_layers=config.locenc_layers,
            num_heads=config.locenc_num_heads,
            num_kv_heads=config.locenc_num_kv_heads,
            patch_size=config.locenc_patch_size,
            use_rope=config.locenc_use_rope,
            rope_theta=config.locenc_rope_theta,
        )
        self.locenc_to_lm_proj = nn.Linear(config.locenc_hidden_dim, config.dim)
        self.lm_speaker_linear = nn.Linear(
            config.lm_speaker_embedding_dim,
            config.dim,
            bias=False,
        )
        self.head = AudioDiTHead(
            latent_dim=config.acoustic_latent_dim,
            cond_dim=config.dim,
            hidden_dim=config.diff_dit_hidden_dim,
            ffn_dim=config.diff_dit_ffn_dim,
            num_layers=config.diff_dit_layers,
            num_heads=config.diff_dit_num_heads,
            num_kv_heads=config.diff_dit_num_kv_heads,
            patch_size=config.diff_dit_patch_size,
            use_rope=config.diff_dit_use_rope,
            rope_theta=config.diff_dit_rope_theta,
            speaker_adaln_zero_enabled=config.diff_dit_speaker_adaln_zero_enabled,
            speaker_embedding_dim=config.diff_dit_speaker_embedding_dim,
            cfg_strength_embedding_enabled=config.diff_dit_cfg_strength_embedding_enabled,
            cfg_strength_max=config.diff_dit_cfg_strength_max,
            step_size_embedding_enabled=config.diff_dit_step_size_embedding_enabled,
            step_schedule=config.diff_dit_step_schedule,
        )
        self.stop_predictor = nn.Linear(config.dim, 2)
        self.register_buffer("speech_scaling_factor", torch.tensor(float("nan")))
        self.register_buffer("speech_bias_factor", torch.tensor(float("nan")))

        keep_layers = int(config.lm_keep_num_hidden_layers)
        config.lm_config.num_hidden_layers = keep_layers
        self.qwen_backbone = AutoModel.from_config(config.lm_config)
        if not hasattr(self.qwen_backbone, "layers"):
            raise ValueError("The Qwen3 backbone does not expose decoder layers.")
        if config.extended_vocab_size:
            self.qwen_backbone.resize_token_embeddings(config.extended_vocab_size)

        backbone_dtype = self.get_input_embeddings().weight.dtype
        self.locenc.to(dtype=backbone_dtype)
        self.locenc_to_lm_proj.to(dtype=backbone_dtype)
        self.lm_speaker_linear.to(dtype=backbone_dtype)
        self.head.to(dtype=torch.float32)
        self.stop_predictor.to(dtype=backbone_dtype)

    def get_input_embeddings(self) -> nn.Module:
        return self.qwen_backbone.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.qwen_backbone.set_input_embeddings(embeddings)

    def forward_lm(
        self,
        inputs_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        return self.qwen_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=bool(output_hidden_states),
            return_dict=True,
        )

    def acoustic_embedding_dtype(self) -> torch.dtype:
        return next(self.locenc.parameters()).dtype

    def embed_acoustic_latents(self, speech_tensor: Tensor) -> Tensor:
        speech_tensor = speech_tensor.to(dtype=self.acoustic_embedding_dtype())
        return self.locenc_to_lm_proj(self.locenc(speech_tensor))

    def forward_speech_features(
        self,
        speech_tensor: Tensor | None,
        mask: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        if speech_tensor is None or speech_tensor.numel() == 0:
            return None, None
        if self.config.scale_acoustic_latent:
            if torch.isnan(self.speech_scaling_factor) or torch.isnan(self.speech_bias_factor):
                raise RuntimeError("The checkpoint is missing acoustic normalization values.")
            speech_tensor = (
                speech_tensor + self.speech_bias_factor
            ) * self.speech_scaling_factor
        return speech_tensor, self.embed_acoustic_latents(speech_tensor)

    def prepare_input_embeds(
        self,
        batch: CuteTTSSegment,
        lm_speaker_embedding: Tensor | None = None,
    ) -> tuple[Tensor, bool, Tensor | None]:
        input_embeds = self.get_input_embeddings()(batch.input_ids)
        contains_speech = batch.speech_tensor.size(1) != 0
        speech_features = None
        if contains_speech:
            speech_features, connected = self.forward_speech_features(
                batch.speech_tensor.type_as(input_embeds),
                batch.speech_pad_mask,
            )
            input_embeds[batch.speech_mask] = connected[batch.speech_pad_mask].to(
                input_embeds.dtype
            )

        if batch.speaker_linear_mask is not None and batch.speaker_linear_mask.any():
            slots = int(batch.speaker_linear_mask.sum().item())
            if lm_speaker_embedding is None:
                lm_speaker_embedding = torch.zeros(
                    slots,
                    self.config.lm_speaker_embedding_dim,
                    device=input_embeds.device,
                )
            if lm_speaker_embedding.size(0) != slots:
                raise ValueError("Speaker embedding rows do not match conditioning slots.")
            projected = self.lm_speaker_linear(
                lm_speaker_embedding.to(
                    device=input_embeds.device,
                    dtype=self.lm_speaker_linear.weight.dtype,
                )
            ).to(input_embeds.dtype)
            input_embeds[batch.speaker_linear_mask.to(input_embeds.device)] = projected
        return input_embeds, contains_speech, speech_features
