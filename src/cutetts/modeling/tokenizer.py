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

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import sentencepiece as spm
from transformers import PreTrainedTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


class CuteTTSSentencePieceTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        sp_model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.vocab_file = str(vocab_file)
        self.sp_model_kwargs = {} if sp_model_kwargs is None else dict(sp_model_kwargs)
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return int(self.sp_model.get_piece_size())

    def get_vocab(self) -> dict[str, int]:
        vocab = {self.sp_model.id_to_piece(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        return list(self.sp_model.encode(text, out_type=str))

    def _convert_token_to_id(self, token: str) -> int:
        token_id = int(self.sp_model.piece_to_id(token))
        return token_id

    def _convert_id_to_token(self, index: int) -> str:
        return str(self.sp_model.id_to_piece(int(index)))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return str(self.sp_model.decode(tokens))

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_name = "tokenizer.model" if filename_prefix is None else f"{filename_prefix}-tokenizer.model"
        out_path = save_dir / out_name
        if Path(self.vocab_file).resolve() != out_path.resolve():
            shutil.copyfile(self.vocab_file, out_path)
        return (str(out_path),)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * (len(token_ids_0) + len(token_ids_1))

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * (len(token_ids_0) + len(token_ids_1))


__all__ = ["CuteTTSSentencePieceTokenizer"]
