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

"""Architecture configuration for the two public CuteTTS variants."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig, Qwen3Config


class CuteTTSConfig(PretrainedConfig):
    model_type = "cutetts"

    def __init__(
        self,
        lm_config: dict[str, Any] | PretrainedConfig | None = None,
        attn_implementation: str = "sdpa",
        torch_dtype: str = "bfloat16",
        lm_keep_num_hidden_layers: int = 7,
        acoustic_latent_dim: int = 64,
        include_semantic_latent: bool = False,
        extended_vocab_size: int = 16385,
        locenc_enabled: bool = True,
        locenc_patch_size: int = 2,
        locenc_use_rope: bool = True,
        locenc_rope_theta: float = 10000.0,
        locenc_layers: int = 2,
        locenc_hidden_dim: int = 1024,
        locenc_ffn_dim: int = 4096,
        locenc_num_heads: int = 16,
        locenc_num_kv_heads: int = 2,
        diff_head_kind: str = "audio_dit",
        diff_dit_patch_size: int = 2,
        diff_dit_use_previous_cond: bool = True,
        diff_dit_use_rope: bool = True,
        diff_dit_rope_theta: float = 10000.0,
        diff_dit_layers: int = 4,
        diff_dit_hidden_dim: int = 1024,
        diff_dit_ffn_dim: int = 4096,
        diff_dit_num_heads: int = 16,
        diff_dit_num_kv_heads: int = 2,
        diff_dit_speaker_adaln_zero_enabled: bool = True,
        diff_dit_speaker_embedding_dim: int = 256,
        diff_dit_cfg_strength_embedding_enabled: bool = False,
        diff_dit_cfg_strength_max: float = 5.0,
        diff_dit_step_size_embedding_enabled: bool = False,
        diff_dit_step_schedule: str = "uniform",
        lm_speaker_linear_enabled: bool = True,
        lm_speaker_embedding_dim: int = 256,
        scale_acoustic_latent: bool = True,
        two_class_stop_predictor: bool = True,
        diff_latent_history: bool = False,
        diff_latent_history_size: int = 0,
        **kwargs,
    ):
        kwargs.pop("use_pretrained_lm", None)
        kwargs.pop("lm_model_name", None)
        super().__init__(**kwargs)
        if lm_config is None:
            lm_config = Qwen3Config()
        self.lm_config = (
            Qwen3Config(**{key: value for key, value in lm_config.items() if key != "model_type"})
            if isinstance(lm_config, dict)
            else lm_config
        )
        self.lm_config._attn_implementation = attn_implementation
        self.attn_implementation = attn_implementation
        self.torch_dtype = torch_dtype
        self.lm_keep_num_hidden_layers = int(lm_keep_num_hidden_layers)
        self.acoustic_latent_dim = int(acoustic_latent_dim)
        self.include_semantic_latent = bool(include_semantic_latent)
        self.extended_vocab_size = int(extended_vocab_size)
        self.locenc_enabled = bool(locenc_enabled)
        self.locenc_patch_size = int(locenc_patch_size)
        self.locenc_use_rope = bool(locenc_use_rope)
        self.locenc_rope_theta = float(locenc_rope_theta)
        self.locenc_layers = int(locenc_layers)
        self.locenc_hidden_dim = int(locenc_hidden_dim)
        self.locenc_ffn_dim = int(locenc_ffn_dim)
        self.locenc_num_heads = int(locenc_num_heads)
        self.locenc_num_kv_heads = int(locenc_num_kv_heads)
        self.diff_head_kind = diff_head_kind
        self.diff_dit_patch_size = int(diff_dit_patch_size)
        self.diff_dit_use_previous_cond = bool(diff_dit_use_previous_cond)
        self.diff_dit_use_rope = bool(diff_dit_use_rope)
        self.diff_dit_rope_theta = float(diff_dit_rope_theta)
        self.diff_dit_layers = int(diff_dit_layers)
        self.diff_dit_hidden_dim = int(diff_dit_hidden_dim)
        self.diff_dit_ffn_dim = int(diff_dit_ffn_dim)
        self.diff_dit_num_heads = int(diff_dit_num_heads)
        self.diff_dit_num_kv_heads = int(diff_dit_num_kv_heads)
        self.diff_dit_speaker_adaln_zero_enabled = bool(diff_dit_speaker_adaln_zero_enabled)
        self.diff_dit_speaker_embedding_dim = int(diff_dit_speaker_embedding_dim)
        self.diff_dit_cfg_strength_embedding_enabled = bool(diff_dit_cfg_strength_embedding_enabled)
        self.diff_dit_cfg_strength_max = float(diff_dit_cfg_strength_max)
        self.diff_dit_step_size_embedding_enabled = bool(diff_dit_step_size_embedding_enabled)
        self.diff_dit_step_schedule = diff_dit_step_schedule
        self.lm_speaker_linear_enabled = bool(lm_speaker_linear_enabled)
        self.lm_speaker_embedding_dim = int(lm_speaker_embedding_dim)
        self.scale_acoustic_latent = bool(scale_acoustic_latent)
        self.two_class_stop_predictor = bool(two_class_stop_predictor)
        self.diff_latent_history = bool(diff_latent_history)
        self.diff_latent_history_size = int(diff_latent_history_size)

    @property
    def dim(self) -> int:
        return int(self.lm_config.hidden_size)
