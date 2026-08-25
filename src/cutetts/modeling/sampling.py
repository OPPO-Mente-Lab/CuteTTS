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

import torch


SAMPLER_COMPILE_MODES = frozenset({"eager", "euler-only", "full-sampler"})
_sampler_compile_mode = "full-sampler"
_compiled_euler_step = None


def set_sampler_compile_mode(mode: str) -> None:
    """Select the production sampler execution mode for this process."""

    if mode not in SAMPLER_COMPILE_MODES:
        raise ValueError(
            f"Unsupported sampler compile mode {mode!r}; expected one of "
            f"{sorted(SAMPLER_COMPILE_MODES)}."
        )
    global _sampler_compile_mode
    _sampler_compile_mode = mode


def get_sampler_compile_mode() -> str:
    return _sampler_compile_mode


def get_score_from_velocity(velocity, x, t):
    # see: https://arxiv.org/pdf/2401.08740
    # page 6, equation (8)
    alpha_t, d_alpha_t = t, 1
    sigma_t, d_sigma_t = 1 - t, -1
    mean = x
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * velocity - mean) / var
    return score


def get_velocity_from_cfg(velocity, cfg, cfg_mult):
    if cfg_mult == 2:
        cond_v, uncond_v = torch.chunk(velocity, 2, dim=0)
        velocity = cond_v + cfg * (cond_v - uncond_v)
    return velocity


def euler_step_eager(x, v, dt: float, cfg: float, cfg_mult: int):
    with torch.amp.autocast("cuda", enabled=False):
        v = v.to(torch.float32)
        v = get_velocity_from_cfg(v, cfg, cfg_mult)
        x = x + v * dt
    return x


def euler_step(x, v, dt: float, cfg: float, cfg_mult: int):
    if _sampler_compile_mode != "euler-only":
        return euler_step_eager(x, v, dt, cfg, cfg_mult)

    global _compiled_euler_step
    if _compiled_euler_step is None:
        _compiled_euler_step = torch.compile(euler_step_eager)
    return _compiled_euler_step(x, v, dt, cfg, cfg_mult)


def euler(
    input_dim,
    forward_fn,
    c,
    cfg: float = 0.0,
    num_sampling_steps: int = 50,
):
    cfg_mult = 1
    if cfg > 0.0:
        cfg_mult = 2

    x_shape = list(c.shape)
    x_shape[0] = x_shape[0] // cfg_mult
    x_shape[-1] = input_dim
    x = torch.randn(x_shape, device=c.device)
    dt = 1.0 / num_sampling_steps
    t = 0
    t_batch = torch.zeros(c.shape[0], device=c.device)
    for _ in range(num_sampling_steps):
        t_batch[:] = t
        combined = torch.cat([x] * cfg_mult, dim=0)
        v = forward_fn(combined, t_batch, c)
        x = euler_step(x, v, dt, cfg, cfg_mult)
        t += dt

    return x
