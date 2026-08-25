# Sway Sampling schedule adapted from F5-TTS:
# https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/cfm.py
#
# Copyright (c) 2024 Yushen CHEN
# Licensed under the MIT License. See LICENSES/F5-TTS-MIT.txt.
#
# Modifications Copyright 2026 OPPO and Fudan University
# Adapted for integration with the CuteTTS Euler sampler.
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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from cutetts.modeling.sampling import euler, get_sampler_compile_mode
from torch import Tensor


def sway_timesteps(
    num_sampling_steps: int,
    coefficient: float,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build F5-TTS Sway Sampling integration boundaries on [0, 1]."""
    if num_sampling_steps <= 0:
        raise ValueError("num_sampling_steps must be positive.")
    upper_bound = 2.0 / (math.pi - 2.0)
    if not -1.0 <= float(coefficient) <= upper_bound:
        raise ValueError(
            "Sway coefficient must be in "
            f"[-1, {upper_bound:.6f}], got {coefficient}."
        )
    u = torch.linspace(
        0.0,
        1.0,
        int(num_sampling_steps) + 1,
        device=device,
        dtype=torch.float32,
    )
    return u + float(coefficient) * (
        torch.cos(0.5 * math.pi * u) - 1.0 + u
    )


def _euler_sway(
    input_dim: int,
    forward_fn,
    condition: Tensor,
    cfg: float,
    *,
    num_sampling_steps: int,
    coefficient: float,
) -> Tensor:
    """Euler flow sampler over the non-uniform F5-TTS Sway time grid."""
    cfg_mult = 2 if cfg > 0.0 else 1
    if condition.size(0) % cfg_mult != 0:
        raise ValueError(
            f"Condition batch {condition.size(0)} is not divisible by CFG multiplier {cfg_mult}."
        )

    sample_shape = list(condition.shape)
    sample_shape[0] = condition.size(0) // cfg_mult
    sample_shape[-1] = input_dim
    x = torch.randn(
        sample_shape,
        device=condition.device,
        dtype=torch.float32,
    )
    times = sway_timesteps(
        num_sampling_steps,
        coefficient,
        device=condition.device,
    )
    t_batch = torch.zeros(condition.size(0), device=condition.device)
    for step in range(int(num_sampling_steps)):
        t_batch[:] = times[step]
        combined_x = torch.cat([x] * cfg_mult, dim=0)
        velocity = forward_fn(combined_x, t_batch, condition).float()
        if cfg_mult == 2:
            conditional, unconditional = torch.chunk(velocity, 2, dim=0)
            velocity = conditional + float(cfg) * (conditional - unconditional)
        dt = times[step + 1] - times[step]
        x = x + velocity * dt
    return x


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_local_rope(query: Tensor, key: Tensor, theta: float) -> tuple[Tensor, Tensor]:
    head_dim = query.size(-1)
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}.")

    seq_len = query.size(-2)
    inv_freq = 1.0 / (
        float(theta)
        ** (
            torch.arange(0, head_dim, 2, device=query.device, dtype=torch.float32)
            / head_dim
        )
    )
    positions = torch.arange(seq_len, device=query.device, dtype=torch.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    view_shape = (1, 1, seq_len, head_dim)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    orig_dtype = query.dtype
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    return (
        ((query * cos) + (_rotate_half(query) * sin)).to(orig_dtype),
        ((key * cos) + (_rotate_half(key) * sin)).to(orig_dtype),
    )


class LocalFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LocalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(f"RoPE requires an even head_dim, got {self.head_dim}.")
            if self.rope_theta <= 0.0:
                raise ValueError("rope_theta must be positive.")
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        bsz, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.use_rope:
            query, key = _apply_local_rope(query, key, self.rope_theta)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
        return self.o_proj(attn_output)


class LocalTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        speaker_adaln_zero_enabled: bool = False,
        speaker_embedding_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_layernorm = nn.RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.self_attn = LocalSelfAttention(
            hidden_dim,
            num_heads,
            num_kv_heads,
            use_rope=use_rope,
            rope_theta=rope_theta,
        )
        self.post_attention_layernorm = nn.RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.mlp = LocalFeedForward(hidden_dim, ffn_dim)
        if speaker_adaln_zero_enabled:
            self.speaker_adaln = nn.Linear(
                int(speaker_embedding_dim),
                6 * self.hidden_dim,
                bias=False,
            )
            nn.init.zeros_(self.speaker_adaln.weight)
        else:
            self.speaker_adaln = None

    @staticmethod
    def _modulate(normed: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, hidden_states: Tensor, speaker_embedding: Tensor | None = None) -> Tensor:
        if self.speaker_adaln is None or speaker_embedding is None:
            hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
            hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
            return hidden_states

        if speaker_embedding.dim() != 2:
            raise ValueError(
                f"speaker_embedding must be [N,C], got {tuple(speaker_embedding.shape)}."
            )
        if speaker_embedding.size(0) != hidden_states.size(0):
            raise ValueError(
                "speaker_embedding batch size must match hidden_states: "
                f"{speaker_embedding.size(0)} vs {hidden_states.size(0)}."
            )
        adaln = self.speaker_adaln(
            speaker_embedding.to(device=hidden_states.device, dtype=hidden_states.dtype)
        )
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = adaln.chunk(6, dim=-1)

        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(self._modulate(normed, shift_attn, scale_attn))
        hidden_states = hidden_states + (1.0 + gate_attn.unsqueeze(1)) * attn_out

        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(self._modulate(normed, shift_mlp, scale_mlp))
        hidden_states = hidden_states + (1.0 + gate_mlp.unsqueeze(1)) * mlp_out
        return hidden_states


class LocalTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        speaker_adaln_zero_enabled: bool = False,
        speaker_embedding_dim: int = 256,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LocalTransformerLayer(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    rms_norm_eps=rms_norm_eps,
                    use_rope=use_rope,
                    rope_theta=rope_theta,
                    speaker_adaln_zero_enabled=speaker_adaln_zero_enabled,
                    speaker_embedding_dim=speaker_embedding_dim,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.RMSNorm(hidden_dim, eps=rms_norm_eps)

    def forward(self, hidden_states: Tensor, speaker_embedding: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, speaker_embedding=speaker_embedding)
        return self.norm(hidden_states)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalPosEmb requires an even dimension.")
        self.dim = int(dim)

    def forward(self, x: Tensor, scale: float = 1000.0) -> Tensor:
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=x.device) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear_1 = nn.Linear(channels, channels, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(channels, channels, bias=True)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class AudioLocEnc(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        num_layers: int = 2,
        num_heads: int = 16,
        num_kv_heads: int = 2,
        patch_size: int = 1,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.special_token = nn.Parameter(torch.randn(1, 1, 1, hidden_dim))
        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=True)
        self.encoder = LocalTransformer(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rms_norm_eps=rms_norm_eps,
            use_rope=use_rope,
            rope_theta=rope_theta,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(2)
        if x.dim() != 4:
            raise ValueError(f"LocEnc expects [B,T,D] or [B,T,P,D], got shape {tuple(x.shape)}.")
        batch_size, num_steps, patch_size, _ = x.shape
        if patch_size != self.patch_size:
            raise ValueError(f"Expected patch_size={self.patch_size}, got {patch_size}.")

        x = self.in_proj(x)
        special_tokens = self.special_token.expand(batch_size, num_steps, 1, -1)
        x = torch.cat([special_tokens, x], dim=2)
        x = x.reshape(batch_size * num_steps, patch_size + 1, x.size(-1))
        hidden_states = self.encoder(x)
        cls_output = hidden_states[:, 0, :]
        return cls_output.reshape(batch_size, num_steps, -1)


class AudioDiTHead(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        cond_dim: int = 1024,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        num_layers: int = 2,
        num_heads: int = 16,
        num_kv_heads: int = 2,
        patch_size: int = 1,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        speaker_adaln_zero_enabled: bool = False,
        speaker_embedding_dim: int = 256,
        cfg_strength_embedding_enabled: bool = False,
        cfg_strength_max: float = 5.0,
        step_size_embedding_enabled: bool = False,
        step_schedule: str = "uniform",
    ):
        super().__init__()
        self.ch_target = int(latent_dim)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.speaker_adaln_zero_enabled = bool(speaker_adaln_zero_enabled)
        self.speaker_embedding_dim = int(speaker_embedding_dim)
        self.cfg_strength_embedding_enabled = bool(cfg_strength_embedding_enabled)
        self.cfg_strength_max = float(cfg_strength_max)
        self.step_size_embedding_enabled = bool(step_size_embedding_enabled)
        self.step_schedule = str(step_schedule)
        if self.step_schedule not in {"uniform", "continuous"}:
            raise ValueError(
                "Standalone inference supports step_schedule='uniform' or 'continuous'."
            )
        if self.cfg_strength_max <= 0.0:
            raise ValueError("cfg_strength_max must be positive.")

        self.in_proj = nn.Linear(latent_dim, hidden_dim, bias=True)
        self.cond_proj = nn.Linear(latent_dim, hidden_dim, bias=True)
        self.mu_proj = nn.Linear(cond_dim, hidden_dim, bias=True) if cond_dim != hidden_dim else nn.Identity()
        self.out_proj = nn.Linear(hidden_dim, latent_dim, bias=True)
        self.time_embeddings = SinusoidalPosEmb(hidden_dim)
        self.time_mlp = TimestepEmbedding(hidden_dim)
        self.delta_time_mlp = TimestepEmbedding(hidden_dim)
        if self.step_size_embedding_enabled:
            self.step_size_embedding = TimestepEmbedding(hidden_dim)
        if self.cfg_strength_embedding_enabled:
            self.cfg_strength_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )
        self.decoder = LocalTransformer(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rms_norm_eps=rms_norm_eps,
            use_rope=use_rope,
            rope_theta=rope_theta,
            speaker_adaln_zero_enabled=self.speaker_adaln_zero_enabled,
            speaker_embedding_dim=self.speaker_embedding_dim,
        )

    def initialize_weights(self) -> None:
        # CuteTTSModel applies the project-wide Linear initialization after this
        # module is constructed, so restore AdaLN-Zero's required zero weights.
        for layer in self.decoder.layers:
            if layer.speaker_adaln is not None:
                nn.init.zeros_(layer.speaker_adaln.weight)
        if self.cfg_strength_embedding_enabled:
            nn.init.zeros_(self.cfg_strength_embedding[-1].weight)
        if self.step_size_embedding_enabled:
            nn.init.zeros_(self.step_size_embedding.linear_2.weight)
            nn.init.zeros_(self.step_size_embedding.linear_2.bias)

    def _step_size_condition(
        self,
        dt: Tensor | float | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        validate: bool = True,
    ) -> Tensor | None:
        if not self.step_size_embedding_enabled:
            return None
        if dt is None:
            raise ValueError("dt is required when step-size embedding is enabled.")
        step_size = torch.as_tensor(dt, device=device, dtype=torch.float32)
        if step_size.ndim == 0:
            step_size = step_size.expand(batch_size)
        step_size = step_size.reshape(-1)
        if step_size.numel() == 1 and batch_size != 1:
            step_size = step_size.expand(batch_size)
        if step_size.numel() != batch_size:
            raise ValueError(
                f"dt must have {batch_size} values, got {step_size.numel()}."
            )
        if validate:
            if not torch.isfinite(step_size).all():
                raise ValueError("dt must contain only finite values.")
            if (step_size <= 0.0).any() or (step_size > 1.0).any():
                raise ValueError("dt must be in (0, 1] for step-distilled inference.")
        step_embedding = self.time_embeddings(step_size).to(dtype=dtype)
        return self.step_size_embedding(step_embedding)

    def _cfg_strength_condition(
        self,
        cfg_strength: Tensor | float | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        validate: bool = True,
    ) -> Tensor | None:
        if not self.cfg_strength_embedding_enabled:
            if cfg_strength is not None:
                raise ValueError(
                    "cfg_strength was provided but cfg strength embedding is disabled."
                )
            return None
        if cfg_strength is None:
            raise ValueError(
                "cfg_strength is required when cfg strength embedding is enabled."
            )
        strength = torch.as_tensor(cfg_strength, device=device, dtype=torch.float32)
        if strength.ndim == 0:
            strength = strength.expand(batch_size)
        strength = strength.reshape(-1)
        if strength.numel() == 1 and batch_size != 1:
            strength = strength.expand(batch_size)
        if strength.numel() != batch_size:
            raise ValueError(
                f"cfg_strength must have {batch_size} values, got {strength.numel()}."
            )
        if validate:
            if not torch.isfinite(strength).all():
                raise ValueError("cfg_strength must contain only finite values.")
            if (strength < 0.0).any() or (strength > self.cfg_strength_max).any():
                raise ValueError(
                    f"cfg_strength must be in [0, {self.cfg_strength_max}]."
                )
        normalized = (strength / self.cfg_strength_max).unsqueeze(-1)
        return self.cfg_strength_embedding(normalized.to(dtype=dtype))

    def _to_patch(self, x: Tensor) -> tuple[Tensor, bool]:
        if x.dim() == 2:
            if self.patch_size != 1:
                if x.size(1) == self.patch_size * self.ch_target:
                    return x.reshape(x.size(0), self.patch_size, self.ch_target), True
                raise ValueError(
                    f"Expected [N,{self.patch_size},{self.ch_target}] or flat [N,{self.patch_size * self.ch_target}], got {tuple(x.shape)}."
                )
            return x.unsqueeze(1), True
        if x.dim() == 3:
            if x.size(1) != self.patch_size:
                raise ValueError(f"Expected patch_size={self.patch_size}, got {x.size(1)}.")
            return x, False
        raise ValueError(f"Expected [N,D] or [N,P,D], got {tuple(x.shape)}.")

    def _predict(
        self,
        x: Tensor,
        t: Tensor,
        z: Tensor,
        cond: Tensor,
        dt: Tensor | None = None,
        speaker_embedding: Tensor | None = None,
        cfg_strength: Tensor | float | None = None,
        validate_conditions: bool = True,
        fixed_condition_cache: list[tuple[Tensor, Tensor, Tensor | None, Tensor | None]] | None = None,
        fixed_condition_index: int | None = None,
    ) -> Tensor:
        x_patch, was_flat = self._to_patch(x)
        cond_patch, _ = self._to_patch(cond)
        if x_patch.shape != cond_patch.shape:
            raise ValueError(f"x and cond shapes must match, got {tuple(x_patch.shape)} and {tuple(cond_patch.shape)}.")
        if z.dim() != 2:
            raise ValueError(f"z must be [N,C], got {tuple(z.shape)}.")
        if speaker_embedding is not None:
            if speaker_embedding.dim() != 2:
                raise ValueError(
                    f"speaker_embedding must be [N,C], got {tuple(speaker_embedding.shape)}."
                )
            if speaker_embedding.size(0) != z.size(0):
                raise ValueError(
                    "speaker_embedding batch size must match z: "
                    f"{speaker_embedding.size(0)} vs {z.size(0)}."
                )

        x_hidden = self.in_proj(x_patch)
        cond_hidden = self.cond_proj(cond_patch)
        prefix = cond_hidden.size(1)

        cached_conditions = None
        if fixed_condition_cache is not None:
            if fixed_condition_index is None:
                raise ValueError(
                    "fixed_condition_index is required with fixed_condition_cache."
                )
            if fixed_condition_index < len(fixed_condition_cache):
                cached_conditions = fixed_condition_cache[fixed_condition_index]

        if cached_conditions is None:
            t_emb = self.time_mlp(
                self.time_embeddings(t).to(dtype=x_hidden.dtype)
            )
            legacy_dt = torch.zeros_like(t) if dt is None else dt
            if self.step_size_embedding_enabled:
                # Keep the legacy branch at dt=0 and add the finite-horizon
                # step as a separate residual.
                legacy_dt = torch.zeros_like(t)
            dt_emb = self.delta_time_mlp(
                self.time_embeddings(legacy_dt).to(dtype=x_hidden.dtype)
            )
        else:
            t_emb, dt_emb, _, _ = cached_conditions
        mu = self.mu_proj(z.to(dtype=x_hidden.dtype))
        if cached_conditions is None:
            strength_condition = self._cfg_strength_condition(
                cfg_strength,
                batch_size=z.size(0),
                device=z.device,
                dtype=x_hidden.dtype,
                validate=validate_conditions,
            )
        else:
            strength_condition = cached_conditions[2]
        if strength_condition is not None:
            mu = mu + strength_condition
        if cached_conditions is None:
            step_size_condition = self._step_size_condition(
                dt,
                batch_size=z.size(0),
                device=z.device,
                dtype=x_hidden.dtype,
                validate=validate_conditions,
            )
        else:
            step_size_condition = cached_conditions[3]
        if step_size_condition is not None:
            mu = mu + step_size_condition
        if fixed_condition_cache is not None and cached_conditions is None:
            if fixed_condition_index != len(fixed_condition_cache):
                raise ValueError("fixed_condition_cache must be populated in order.")
            fixed_condition_cache.append(
                (t_emb, dt_emb, strength_condition, step_size_condition)
            )
        seq = torch.cat([(mu + t_emb + dt_emb).unsqueeze(1), cond_hidden, x_hidden], dim=1)
        hidden_states = self.decoder(seq, speaker_embedding=speaker_embedding)
        output = self.out_proj(hidden_states[:, prefix + 1 :, :])
        return output[:, 0, :] if was_flat else output

    def _euler_sample_core(
        self,
        x: Tensor,
        z: Tensor,
        cond: Tensor,
        speaker_embedding: Tensor | None,
        distilled_cfg_strength: Tensor | float | None,
        cfg: float,
        num_sampling_steps: int,
    ) -> Tensor:
        """Fixed-shape DiT + Euler loop used by full-sampler compilation."""

        cfg_mult = 2 if cfg > 0.0 else 1
        sample_dim = self.ch_target * self.patch_size
        step_dt = 1.0 / int(num_sampling_steps)
        distilled_dt = step_dt if self.step_size_embedding_enabled else None
        t_batch = torch.zeros(z.size(0), device=z.device, dtype=torch.float32)

        for step in range(int(num_sampling_steps)):
            t_batch.fill_(step * step_dt)
            combined = torch.cat([x] * cfg_mult, dim=0)
            model_dt = (
                None
                if distilled_dt is None
                else torch.full_like(t_batch, distilled_dt)
            )
            if self.patch_size == 1:
                velocity = self._predict(
                    combined,
                    t_batch,
                    z,
                    cond.to(dtype=combined.dtype),
                    dt=model_dt,
                    speaker_embedding=speaker_embedding,
                    cfg_strength=distilled_cfg_strength,
                    validate_conditions=False,
                )
            else:
                combined_patch = combined.reshape(
                    combined.size(0),
                    self.patch_size,
                    self.ch_target,
                )
                velocity = self._predict(
                    combined_patch,
                    t_batch,
                    z,
                    cond.to(dtype=combined.dtype),
                    dt=model_dt,
                    speaker_embedding=speaker_embedding,
                    cfg_strength=distilled_cfg_strength,
                    validate_conditions=False,
                ).reshape(combined.size(0), sample_dim)

            velocity = velocity.to(torch.float32)
            if cfg_mult == 2:
                conditional, unconditional = torch.chunk(velocity, 2, dim=0)
                velocity = conditional + cfg * (conditional - unconditional)
            x = x + velocity * step_dt

        return x

    def _run_compiled_euler_sample(
        self,
        x: Tensor,
        z: Tensor,
        cond: Tensor,
        speaker_embedding: Tensor | None,
        distilled_cfg_strength: Tensor | float | None,
        cfg: float,
        num_sampling_steps: int,
    ) -> Tensor:
        compiled = self.__dict__.get("_compiled_euler_sample_core")
        if compiled is None:
            compiled = torch.compile(
                self._euler_sample_core,
                fullgraph=True,
                mode="default",
            )
            object.__setattr__(self, "_compiled_euler_sample_core", compiled)
        return compiled(
            x,
            z,
            cond,
            speaker_embedding,
            distilled_cfg_strength,
            cfg,
            num_sampling_steps,
        )

    def _prepare_speaker_for_sample(
        self,
        z: Tensor,
        speaker_embedding: Tensor | None,
        cfg_mult: int,
    ) -> Tensor | None:
        if speaker_embedding is None:
            return None
        if speaker_embedding.dim() != 2:
            raise ValueError(
                f"speaker_embedding must be [N,C], got {tuple(speaker_embedding.shape)}."
            )
        batch_size = z.size(0) // cfg_mult
        if speaker_embedding.size(0) == batch_size and cfg_mult > 1:
            speaker_embedding = speaker_embedding.repeat(cfg_mult, 1)
        if speaker_embedding.size(0) != z.size(0):
            raise ValueError(
                "speaker_embedding batch size does not match z after cfg repeat: "
                f"{speaker_embedding.size(0)} vs {z.size(0)}."
            )
        return speaker_embedding.to(device=z.device)

    def _prepare_cond_for_sample(self, z: Tensor, cond: Tensor | None, cfg_mult: int) -> Tensor:
        batch_size = z.size(0) // cfg_mult
        if cond is None:
            cond = z.new_zeros((batch_size, self.ch_target))
            if self.patch_size != 1:
                cond = z.new_zeros((batch_size, self.patch_size, self.ch_target))
        if cond.dim() == 2 and self.patch_size != 1:
            if cond.size(1) != self.patch_size * self.ch_target:
                raise ValueError(
                    f"cond must be [N,{self.patch_size},{self.ch_target}] or flat [N,{self.patch_size * self.ch_target}], got {tuple(cond.shape)}."
                )
            cond = cond.reshape(cond.size(0), self.patch_size, self.ch_target)
        if cond.dim() == 3 and self.patch_size == 1:
            cond = cond[:, 0, :]
        expected_dim = 2 if self.patch_size == 1 else 3
        if cond.dim() != expected_dim:
            raise ValueError(f"cond has unexpected shape for patch_size={self.patch_size}: {tuple(cond.shape)}.")
        if cond.size(0) == batch_size and cfg_mult > 1:
            repeat_shape = (cfg_mult,) + (1,) * (cond.dim() - 1)
            cond = cond.repeat(*repeat_shape)
        if cond.size(0) != z.size(0):
            raise ValueError(f"cond batch size {cond.size(0)} does not match z batch size {z.size(0)}.")
        return cond.to(device=z.device)

    def sample(
        self,
        z: Tensor,
        cfg: float,
        num_sampling_steps: int,
        cond: Tensor | None = None,
        speaker_embedding: Tensor | None = None,
        sway_sampling_coefficient: float = 0.0,
        distilled_cfg_strength: Tensor | float | None = None,
        sampling_condition_cache: list[tuple[Tensor, Tensor, Tensor | None, Tensor | None]] | None = None,
    ) -> Tensor:
        if self.step_size_embedding_enabled:
            if int(num_sampling_steps) <= 0:
                raise ValueError("num_sampling_steps must be positive.")
            if (
                self.step_schedule != "continuous"
                and int(num_sampling_steps) not in {1, 2, 4}
            ):
                raise ValueError(
                    "Step-distilled Audio DiT supports 1, 2, or 4 sampling steps."
                )
            if cfg != 0.0:
                raise ValueError(
                    "Step-distilled Audio DiT expects single-branch distilled CFG."
                )
            if sway_sampling_coefficient != 0.0:
                raise ValueError(
                    "Step-distilled Audio DiT does not support Sway Sampling."
                )
        cfg_mult = 2 if cfg > 0.0 else 1
        cond = self._prepare_cond_for_sample(z, cond, cfg_mult)
        speaker_embedding = self._prepare_speaker_for_sample(
            z,
            speaker_embedding,
            cfg_mult,
        )
        sample_dim = self.ch_target * self.patch_size
        distilled_dt = (
            1.0 / int(num_sampling_steps)
            if self.step_size_embedding_enabled
            else None
        )
        if sampling_condition_cache is not None:
            if not self.step_size_embedding_enabled:
                raise ValueError(
                    "sampling_condition_cache is only supported by step-distilled Euler sampling."
                )
            if len(sampling_condition_cache) not in {0, int(num_sampling_steps)}:
                raise ValueError(
                    "sampling_condition_cache length must be zero or num_sampling_steps."
                )

        if (
            get_sampler_compile_mode() == "full-sampler"
            and sway_sampling_coefficient == 0.0
        ):
            # Keep value-dependent validation outside the compiled graph. The
            # eager implementation performs these checks from _predict() on
            # every step; a fixed sampler only needs to validate once.
            if self.cfg_strength_embedding_enabled:
                strength = torch.as_tensor(
                    distilled_cfg_strength,
                    device=z.device,
                    dtype=torch.float32,
                ).reshape(-1)
                if not torch.isfinite(strength).all():
                    raise ValueError("cfg_strength must contain only finite values.")
                if (strength < 0.0).any() or (strength > self.cfg_strength_max).any():
                    raise ValueError(
                        f"cfg_strength must be in [0, {self.cfg_strength_max}]."
                    )
            if distilled_dt is not None and not 0.0 < distilled_dt <= 1.0:
                raise ValueError("dt must be in (0, 1] for step-distilled inference.")

            x_shape = list(z.shape)
            x_shape[0] = z.size(0) // cfg_mult
            x_shape[-1] = sample_dim
            initial_x = torch.randn(x_shape, device=z.device)
            output = self._run_compiled_euler_sample(
                initial_x,
                z,
                cond,
                speaker_embedding,
                distilled_cfg_strength,
                cfg,
                num_sampling_steps,
            )
            if self.patch_size != 1:
                output = output.reshape(
                    output.size(0),
                    self.patch_size,
                    self.ch_target,
                )
            return output

        fixed_condition_index = 0

        def forward_fn(x: Tensor, t: Tensor, c: Tensor) -> Tensor:
            nonlocal fixed_condition_index
            dt = (
                None
                if distilled_dt is None
                else torch.full_like(t, distilled_dt)
            )
            condition_index = fixed_condition_index
            fixed_condition_index += 1
            if self.patch_size == 1:
                return self._predict(
                    x,
                    t,
                    c,
                    cond.to(dtype=x.dtype),
                    dt=dt,
                    speaker_embedding=speaker_embedding,
                    cfg_strength=distilled_cfg_strength,
                    fixed_condition_cache=sampling_condition_cache,
                    fixed_condition_index=condition_index,
                )
            x_patch = x.reshape(x.size(0), self.patch_size, self.ch_target)
            output = self._predict(
                x_patch,
                t,
                c,
                cond.to(dtype=x.dtype),
                dt=dt,
                speaker_embedding=speaker_embedding,
                cfg_strength=distilled_cfg_strength,
                fixed_condition_cache=sampling_condition_cache,
                fixed_condition_index=condition_index,
            )
            return output.reshape(output.size(0), sample_dim)

        if sway_sampling_coefficient != 0.0:
            output = _euler_sway(
                sample_dim,
                forward_fn,
                z,
                cfg,
                num_sampling_steps=num_sampling_steps,
                coefficient=sway_sampling_coefficient,
            )
            if self.patch_size != 1:
                output = output.reshape(output.size(0), self.patch_size, self.ch_target)
            return output
        output = euler(
            sample_dim,
            forward_fn,
            z,
            cfg,
            num_sampling_steps=num_sampling_steps,
        )
        if self.patch_size != 1:
            output = output.reshape(output.size(0), self.patch_size, self.ch_target)
        return output
