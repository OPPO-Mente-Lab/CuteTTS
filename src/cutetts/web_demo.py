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

"""Launch the local CuteTTS streaming web demo."""

from __future__ import annotations

import argparse
import sysconfig
from pathlib import Path

import uvicorn

from cutetts.demo.app import create_app
from cutetts.demo.config import DemoSettings
from cutetts.runtime import resolve_device


def _default_warmup_reference() -> Path:
    candidates = (
        Path(__file__).resolve().parents[2] / "assets" / "default_reference.wav",
        Path(sysconfig.get_path("data"))
        / "share"
        / "cutetts"
        / "default_reference.wav",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the CuteTTS web demo.")
    parser.add_argument(
        "--model-dir",
        default="model",
        help="Directory containing CuteTTS and CuteTTS-distill (default: ./model).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Initial device selection shown in the web UI (default: auto).",
    )
    parser.add_argument("--warmup-reference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="CPU intra-op threads (default: 4 with MPS, otherwise 16).",
    )
    parser.add_argument("--cpu-interop-threads", type=int, default=1)
    return parser


def build_settings(args: argparse.Namespace) -> DemoSettings:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535].")
    cpu_threads = args.cpu_threads
    if cpu_threads is None:
        cpu_threads = 4 if resolve_device(args.device).type == "mps" else 16
    if cpu_threads <= 0 or args.cpu_interop_threads <= 0:
        raise ValueError("CPU thread counts must be positive.")
    warmup_reference = (
        Path(args.warmup_reference).expanduser().resolve()
        if args.warmup_reference
        else _default_warmup_reference()
    )
    return DemoSettings(
        model_root=Path(args.model_dir).expanduser().resolve(),
        warmup_reference=warmup_reference,
        initial_device=args.device,
        host=args.host,
        port=args.port,
        cpu_threads=cpu_threads,
        cpu_interop_threads=args.cpu_interop_threads,
    )


def main() -> None:
    settings = build_settings(build_parser().parse_args())
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
