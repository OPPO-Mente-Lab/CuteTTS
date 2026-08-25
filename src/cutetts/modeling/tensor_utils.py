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
from torch import Tensor, LongTensor
from typing import Any, List, Union, Tuple


class TensorStack:
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, xs: List[Tensor]) -> Tensor:
        if xs[0].shape[0] == 1:
            return torch.cat(xs, dim=0)
        else:
            return torch.stack(xs, dim=0)


class TensorPad:
    def __init__(self, pad_with=0, return_length: bool = False, length_idx: int = 0) -> None:
        self.pad_width = pad_with
        self.return_length = return_length
        self.length_idx = length_idx

    def __call__(
        self,
        xs: List[Tensor],
    ):
        return self.pad_tensors(xs)

    def pad_tensors(self, xs: List[Tensor]) -> Union[Tuple[Tensor, Tensor], Tensor]:
        bcsz = len(xs)
        lengths = LongTensor([x.size(self.length_idx) for x in xs])

        _example = xs[0]
        padded_buffer = _example.data.new_ones((bcsz,) + self.max_size_by_dim(xs)) * self.pad_width

        for batch_idx in range(bcsz):
            piece = xs[batch_idx]
            piece_index = self.get_piece_index(batch_idx, piece.shape)
            padded_buffer[piece_index] = piece

        if self.return_length:
            return padded_buffer, lengths.to(padded_buffer.device)
        else:
            return padded_buffer

    # for multi-dimension padding in dataloader
    @staticmethod
    def max_size_by_dim(batch_col: List[Tensor]) -> Tuple[int, ...]:
        """
        return the maximum length along all dims in a list of tensors
        (have to be matching n_dims or assertion error)
        """

        def assert_align(all_sizes):
            n_dims = Tensor([len(s) for s in all_sizes])
            assert (n_dims - n_dims[0] == 0).all(), f"batch column not aligned : {n_dims}"
            return n_dims[0]

        all_sizes = [b.shape for b in batch_col]
        n_dim = assert_align(all_sizes)
        return tuple(max(lengths_at_idx) for lengths_at_idx in zip(*all_sizes))

    @staticmethod
    def get_piece_index(batch_idx, piece_shape):
        """
        return a piece's index (range) in the padded buffer (with batch-index as a parameter)
        """
        return tuple([batch_idx] + [slice(piece_len) for piece_len in piece_shape])


def NoOp(batch):
    return batch


class TupleCollatorBuilder:
    def __init__(self, *collators):
        self.collators = collators

    def __call__(self, raw_batch) -> Any:
        return tuple(collate_fn(sub_batch) for collate_fn, sub_batch in zip(self.collators, zip(*raw_batch)))
