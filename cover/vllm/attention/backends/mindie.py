# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# Part of codes in this file was copied from project [vLLM Team][vllm]

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Type

import torch
from atb_llm.utils.initial import NPUSocInfo
from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata, AttentionMetadataBuilder

if TYPE_CHECKING:
    from vllm.worker.npu_model_runner import ModelInputForNPUBuilder, ModelInputForNPUWithSamplingMetadata

from vllm.attention.backends.utils import (
    PAD_SLOT_ID,
    CommonAttentionState,
    compute_slot_mapping,
    compute_slot_mapping_start_idx,
    is_block_tables_empty,
)
from vllm.utils import async_tensor_h2d, make_tensor_with_pad


class MindIEAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "mindie-attn-backend"

    @staticmethod
    def get_impl_cls():
        return None

    @staticmethod
    def get_metadata_cls() -> Type["MindIEAttentionMetadata"]:
        return MindIEAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["MindIEAttentionMetadataBuilder"]:
        return MindIEAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if not NPUSocInfo().need_nz:
            return (num_blocks, block_size, num_kv_heads, head_size)
        else:
            return (num_blocks, num_kv_heads * head_size // 16, block_size, 16)

    @staticmethod
    def swap_blocks(src_kv_cache: torch.Tensor, dst_kv_cache: torch.Tensor, src_to_dst: torch.Tensor) -> None: ...

    @staticmethod
    def copy_blocks(kv_caches: List[torch.Tensor], src_to_dists: torch.Tensor) -> None: 
        for pair in src_to_dists:
            src, dst = pair.tolist()  # Convert tensor elements to Python integers
            for key_cache, value_cache in kv_caches:
                key_cache.data[dst, :] = key_cache.data[src, :]
                value_cache.data[dst, :] = value_cache.data[src, :]


@dataclass
class MindIEAttentionMetadata(AttentionMetadata):
    """Metadata for AscendAttentionBackend."""

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # Maximum query length in the batch.
    max_query_len: Optional[int]
    # Maximum sequence length in the batch.
    max_seq_len: Optional[int]
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    use_cuda_graph: bool

    _cached_prefill_metadata: Optional["MindIEAttentionMetadata"] = None
    _cached_decode_metadata: Optional["MindIEAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["MindIEAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens_tensor is not None
        assert self.block_tables is not None
        assert self.seq_start_loc is not None

        self._cached_prefill_metadata = MindIEAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[: self.num_prefill_tokens],
            seq_lens=self.seq_lens[: self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[: self.num_prefills],
            max_query_len=self.max_query_len,
            max_seq_len=max(self.seq_lens),
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[: self.num_prefills + 1],
            seq_start_loc=self.seq_start_loc[: self.num_prefills + 1],
            context_lens_tensor=self.context_lens_tensor[: self.num_prefills],
            block_tables=self.block_tables[: self.num_prefills],
            use_cuda_graph=False,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["MindIEAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = MindIEAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens :],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills :],
            max_query_len=None,
            max_seq_len=max(self.seq_lens),
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills :],
            use_cuda_graph=self.use_cuda_graph,
        )
        return self._cached_decode_metadata

    def advance_step(
        self,
        model_input: "ModelInputForNPUWithSamplingMetadata",
        sampled_token_ids: Optional[torch.Tensor],
        block_size: int,
        num_seqs: int,
        num_queries: int,
    ):
        """
        Update metadata in-place to advance one decode step.
        """
        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs,)

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs,)
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0
        assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1,)
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1,)

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries,)

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        advance_step_flashattn(
            num_seqs=num_seqs,
            num_queries=num_queries,
            block_size=block_size,
            input_tokens=model_input.input_tokens,
            sampled_token_ids=sampled_token_ids,
            input_positions=model_input.input_positions,
            seq_lens=self.seq_lens_tensor,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables,
            block_tables_stride=self.block_tables.stride(0),
        )


def advance_step_flashattn(
    num_seqs,
    num_queries,
    block_size,
    input_tokens,
    sampled_token_ids,
    input_positions,
    seq_lens,
    slot_mapping,
    block_tables,
    block_tables_stride,
):

    # Update input_tokens: matching the shape of input_tokens and sampled_token_ids
    input_tokens[:num_queries] = sampled_token_ids[:num_queries].squeeze(1)

    # Update sequence lengths and input positions
    next_seq_len = seq_lens[:num_queries] + 1
    next_input_pos = next_seq_len - 1
    seq_lens[:num_queries] = next_seq_len
    input_positions[:num_queries] = next_input_pos

    # Compute block indices and offsets
    block_index = next_input_pos // block_size
    block_offset = next_input_pos % block_size

    # Retrieve sequence-specific block tables
    seq_block_tables = block_tables[:num_queries, :block_tables_stride]

    # Use gather to map block indices to slots
    slot_num = seq_block_tables.gather(1, block_index.unsqueeze(1)).squeeze(1) * block_size + block_offset
    slot_mapping[:num_queries] = slot_num


class MindIEAttentionMetadataBuilder(AttentionMetadataBuilder[MindIEAttentionMetadata]):

    def __init__(self, input_builder: "ModelInputForNPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.use_v2_block_manager = input_builder.scheduler_config.use_v2_block_manager

    def build(self, seq_lens: List[int], query_lens: List[int], cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any([inter_data.prefix_cache_hit for inter_data in self.input_builder.inter_data_list])
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data, self.input_builder.chunked_prefill_enabled, prefix_cache_hit)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        max_seq_len = max(seq_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int, device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device, self.runner.pin_memory)
        query_lens_tensor = async_tensor_h2d(query_lens, torch.long, device, self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long, device, self.runner.pin_memory)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1, dtype=torch.int32, device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1, dtype=torch.int32, device=device)
        torch.cumsum(seq_lens_tensor, dim=0, dtype=seq_start_loc.dtype, out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor, dim=0, dtype=query_start_loc.dtype, out=query_start_loc[1:])

        # TODO: Remove the unnecessary params
        return MindIEAttentionMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )

    def _add_seq_group(
        self,
        inter_data: "ModelInputForNPUBuilder.InterDataForSeqGroup",
        chunked_prefill_enabled: bool,
        prefix_cache_hit: bool,
    ):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for seq_id, token_len, seq_len, curr_seq_len, query_len, context_len, _ in zip(
            inter_data.seq_ids,
            [len(t) for t in inter_data.input_tokens],
            inter_data.orig_seq_lens,
            inter_data.seq_lens,
            inter_data.query_lens,
            inter_data.context_lens,
            inter_data.curr_sliding_window_blocks,
        ):
            self.context_lens.append(context_len)

            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, "seq_len: {}, context_len: {}, query_len: {}".format(
                    seq_len, context_len, query_len
                )
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []

            # Adapt for prefix-cahce
            if inter_data.block_tables:
                block_table = inter_data.block_tables[seq_id]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window, self.use_v2_block_manager
            )
            compute_slot_mapping(
                is_profile_run,
                self.slot_mapping,
                seq_id,
                seq_len,
                context_len,
                start_idx,
                self.block_size,
                inter_data.block_tables,
            )