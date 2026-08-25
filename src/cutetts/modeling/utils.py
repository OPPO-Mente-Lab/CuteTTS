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

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from cutetts.modeling.tensor_utils import TensorPad
from loguru import logger
from torch import Tensor


@dataclass
class BatchExtractor:
    """Batch feature extraction for nested lists of optional tensors.

    The extractor flattens and pads the tensors, optionally sorts and splits
    them into smaller batches, runs ``model_forward``, and restores the original
    nested layout. ``None`` entries retain their positions.
    """

    input_length_dim: int
    output_length_dim: int
    length_scale: float
    internal_batch_size: Optional[int]
    sort_batches: bool = False
    provide_lengths: bool = False

    def __post_init__(self, ):
        self.tensor_padder = TensorPad(pad_with=0, return_length=self.provide_lengths, length_idx=self.input_length_dim)

    def restore_batched_tensor(self, batched_tensor: Tensor, original_lengths: "list[int]") -> Tensor:
        result = []
        for output_tensor, orig_length in zip(batched_tensor, original_lengths):
            output_length = math.ceil(orig_length * self.length_scale)

            if output_length <= output_tensor.shape[self.output_length_dim]:
                sliced_tensor = output_tensor.narrow(self.output_length_dim, 0, output_length)
            else:
                pad_size = output_length - output_tensor.shape[self.output_length_dim]
                pad_shape = list(output_tensor.shape)
                pad_shape[self.output_length_dim] = pad_size
                padding = torch.zeros(pad_shape, dtype=output_tensor.dtype, device=output_tensor.device)
                sliced_tensor = torch.cat([output_tensor, padding], dim=self.output_length_dim)
            result.append(sliced_tensor)
        return result

    def __call__(self, nested_batch: "list[list[Tensor | None]]", model_forward: Callable[[Tensor], Tensor]) -> "list[list[Tensor | None]]":
        """Extract a padded batch and restore the original nested structure."""

        flat_tensors = []
        positions = []
        original_lengths = []

        for i, inner_list in enumerate(nested_batch):
            for j, tensor in enumerate(inner_list):
                if tensor is not None:
                    flat_tensors.append(tensor)
                    positions.append((i, j))
                    original_lengths.append(tensor.shape[self.input_length_dim])

        if not flat_tensors:
            return nested_batch

        if self.sort_batches:
            sort_indices = sorted(range(len(flat_tensors)),
                                key=lambda idx: original_lengths[idx],
                                reverse=True)
        else:
            sort_indices = list(range(len(flat_tensors)))

        sorted_tensors = [flat_tensors[i] for i in sort_indices]
        sorted_lengths = [original_lengths[i] for i in sort_indices]
        sorted_positions = [positions[i] for i in sort_indices]

        total_compute = 0
        output_tensors = []

        internal_batch_size = self.internal_batch_size
        if internal_batch_size is None or internal_batch_size <= 0:
            internal_batch_size = len(sorted_tensors)

        if len(sorted_tensors) > internal_batch_size:
            for i in range(0, len(sorted_tensors), internal_batch_size):
                batch_slice = sorted_tensors[i: i+internal_batch_size]
                if self.provide_lengths:
                    batch_slice, batch_lengths = self.tensor_padder.pad_tensors(batch_slice)
                    assert isinstance(batch_slice, Tensor)
                    output_slice = model_forward(batch_slice, batch_lengths).clone()
                else:
                    batch_slice = self.tensor_padder.pad_tensors(batch_slice)
                    assert isinstance(batch_slice, Tensor)
                    output_slice = model_forward(batch_slice).clone()

                total_compute += batch_slice.size(0) * batch_slice.size(self.input_length_dim + 1)
                output_tensors += self.restore_batched_tensor(output_slice, sorted_lengths[i: i+internal_batch_size])
        else:
            if self.provide_lengths:
                batch, batch_lengths = self.tensor_padder.pad_tensors(sorted_tensors)
                assert isinstance(batch, Tensor)
                output_batch = model_forward(batch, batch_lengths).clone()
            else:
                batch = self.tensor_padder.pad_tensors(sorted_tensors)
                assert isinstance(batch, Tensor)
                output_batch = model_forward(batch).clone()
            total_compute += batch.size(0) * batch.size(self.input_length_dim + 1)
            output_tensors += self.restore_batched_tensor(output_batch, sorted_lengths)

        sorted_output = output_tensors
        original_order_output = [None] * len(sorted_output)
        for sorted_idx, original_idx in enumerate(sort_indices):
            original_order_output[original_idx] = sorted_output[sorted_idx]

        result = [[None for _ in inner_list] for inner_list in nested_batch]

        for tensor, (i, j) in zip(original_order_output, positions):
            result[i][j] = tensor

        logger.debug(f'[batch-extractor] #compute: {total_compute / 1e3}k')

        return result
