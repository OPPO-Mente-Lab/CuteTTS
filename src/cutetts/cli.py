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

"""Command-line interface for one-shot synthesis."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from cutetts import CuteTTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate speech with CuteTTS.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--mode", choices=("tts", "voice_clone"), default="tts")
    parser.add_argument("--reference-audio")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cuda, mps, or cpu (default: auto).",
    )
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--diffusion-steps", type=int)
    parser.add_argument("--diffusion-sway-coefficient", type=float)
    parser.add_argument("--max-decode-length", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = CuteTTS.from_pretrained(args.model_dir, device=args.device)
    result = model.generate(
        args.text,
        mode=args.mode,
        reference_audio=args.reference_audio,
        cfg_strength=args.cfg_strength,
        diffusion_steps=args.diffusion_steps,
        diffusion_sway_coefficient=args.diffusion_sway_coefficient,
        max_decode_length=args.max_decode_length,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    waveform = result.waveform.squeeze(0).float().numpy()
    sf.write(output, waveform, result.sample_rate)
    print(output)


if __name__ == "__main__":
    main()
