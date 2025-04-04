# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# Part of code in this file was copied from project [vLLM Team][vllm] for adapting usage

import dataclasses
import functools
import itertools
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import torch
import torch.distributed
import torch.nn as nn
import vllm.envs as envs
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    PromptAdapterConfig,
    SchedulerConfig,
)
from vllm.core.scheduler import SchedulerOutputs
from vllm.distributed import get_pp_group
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata, SamplingMetadataCache
from vllm.model_executor.layers.sampler import (
    PromptLogprobs,
    SampleLogprobs,
    SamplerOutput,
    SamplingMetadata,
    get_logprobs,
    get_pythonized_sample_results,
)
from vllm.model_executor.model_loader.npu import get_model
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models.interfaces import supports_lora, supports_multimodal
from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
from vllm.multimodal import MULTIMODAL_REGISTRY, BatchedTensorInputs, MultiModalInputs, MultiModalRegistry
from vllm.prompt_adapter.layers import PromptAdapterMapping
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.prompt_adapter.worker_manager import LRUCacheWorkerPromptAdapterManager
from vllm.sampling_params import SamplingParams
from vllm.sequence import (
    CompletionSequenceGroupOutput,
    IntermediateTensors,
    Logprob,
    SequenceGroupMetadata,
    SequenceOutput,
)
from vllm.utils import DeviceMemoryProfiler, PyObjectCache, async_tensor_h2d, flatten_2d_lists, is_pin_memory_available
from vllm.worker.model_runner_base import (
    BroadcastableModelInput,
    ModelRunnerBase,
    ModelRunnerInputBase,
    ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_frozen_model_input_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict,
)

from ..model_executor.model_loader.tensorizer import TensorizerConfig

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

LORA_WARMUP_RANK = 8
_BATCH_SIZE_ALIGNMENT = 8
# all the token sizes that **can** be captured by cudagraph.
# they can be arbitrarily large.
# currently it includes: 1, 2, 4, 8, 16, 24, 32, 40, ..., 8192.
# the actual sizes to capture will be determined by the model,
# depending on the model's max_num_seqs.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [_BATCH_SIZE_ALIGNMENT * i for i in range(1, 1025)]
MULTI_STEP_ATTENTION_BACKENDS = ["flash-attn", "rocm-flash-attn", "flashinfer", "mindie-attn-backend"]

TModelInputForNPU = TypeVar("TModelInputForNPU", bound="ModelInputForNPU")


@dataclass(frozen=True)
class ModelInputForNPU(ModelRunnerInputBase):
    """
    This base class contains metadata needed for the base model forward pass
    but not metadata for possible additional steps, e.g., sampling. Model
    runners that run additional steps should subclass this method to add
    additional fields.
    """

    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    lora_mapping: Optional["LoRAMapping"] = None
    lora_requests: Optional[List[LoRARequest]] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    prompt_adapter_mapping: Optional[PromptAdapterMapping] = None
    prompt_adapter_requests: Optional[Set[PromptAdapterRequest]] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    request_ids_to_seq_ids: Optional[Dict[str, List[int]]] = None
    finished_requests_ids: Optional[List[str]] = None
    virtual_engine: int = 0
    async_callback: Optional[Callable] = None
    seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None
    scheduler_outputs: Optional[SchedulerOutputs] = None

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForNPU],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForNPU:
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(attn_backend, tensor_dict)
        return cls(**tensor_dict)

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "prompt_adapter_mapping": self.prompt_adapter_mapping,
            "prompt_adapter_requests": self.prompt_adapter_requests,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict


@dataclass(frozen=True)
class ModelInputForNPUWithSamplingMetadata(ModelInputForNPU):
    """
    Used by the ModelRunner.
    """

    sampling_metadata: Optional["SamplingMetadata"] = None
    # Used for speculative decoding. We do not broadcast it because it is only
    # used by the driver worker.
    is_prompt: Optional[bool] = None

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForNPUWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(attn_backend, tensor_dict)
        return cls(**tensor_dict)

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "prompt_adapter_mapping": self.prompt_adapter_mapping,
            "prompt_adapter_requests": self.prompt_adapter_requests,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict, self.sampling_metadata)
        return tensor_dict


class ModelInputForNPUBuilder(ModelRunnerInputBuilderBase[ModelInputForNPU]):
    """Build ModelInputForNPU from SequenceGroupMetadata."""

    # Note: ideally we would be using a dataclass(kw_only=True)
    # here, so that this can be subclassed easily,
    # but kw_only is not supported in python<3.10.
    class InterDataForSeqGroup:
        """Intermediate data for the current sequence group."""

        def __init__(
            self,
            *,
            # From sequence group metadata.
            request_id: str,
            seq_ids: List[int],
            is_prompt: bool,
            block_tables: Optional[Dict[int, List[int]]],
            computed_block_nums: List[int],
            n_seqs: int = 0,
            # Input tokens and positions.
            input_tokens: Optional[List[List[int]]] = None,
            input_positions: Optional[List[List[int]]] = None,
            # The sequence length (may be capped to the sliding window).
            seq_lens: Optional[List[int]] = None,
            # The original sequence length (before applying sliding window).
            # This is used to compute slot mapping.
            orig_seq_lens: Optional[List[int]] = None,
            # The query length.
            query_lens: Optional[List[int]] = None,
            # The number of tokens that are already computed.
            context_lens: Optional[List[int]] = None,
            # The current sliding window block.
            curr_sliding_window_blocks: Optional[List[int]] = None,
            # LoRA inputs.
            lora_index_mapping: Optional[List[List[int]]] = None,
            lora_prompt_mapping: Optional[List[List[int]]] = None,
            lora_requests: Optional[List[LoRARequest]] = None,
            # Prompt adapter inputs.
            prompt_adapter_index_mapping: Optional[List[int]] = None,
            prompt_adapter_prompt_mapping: Optional[List[int]] = None,
            prompt_adapter_request: Optional[PromptAdapterRequest] = None,
            # Multi-modal inputs.
            multi_modal_inputs: Optional[MultiModalInputs] = None,
            # Whether the prefix cache is hit (prefill only).
            prefix_cache_hit: bool = False,
            reinit: bool = False,
            reinit_use_defaults: bool = False,
        ):
            if reinit:
                assert len(self.seq_ids) == len(seq_ids)  # type: ignore
                for i, seq_id in enumerate(seq_ids):
                    self.seq_ids[i] = seq_id  # type: ignore
            else:
                self.seq_ids = seq_ids

            self.request_id = request_id
            self.is_prompt = is_prompt
            self.block_tables = block_tables
            self.computed_block_nums = computed_block_nums
            self.n_seqs = n_seqs

            if reinit:
                if len(self.seq_ids) == 1 and reinit_use_defaults:
                    self.simple_reinit()
                else:
                    if input_tokens:
                        self.input_tokens = input_tokens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.input_tokens[seq_id].clear()

                    if input_positions:
                        self.input_positions = input_positions
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.input_positions[seq_id].clear()

                    if seq_lens:
                        self.seq_lens = seq_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.seq_lens[seq_id] = 0

                    if orig_seq_lens:
                        self.orig_seq_lens = orig_seq_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.orig_seq_lens[seq_id] = 0

                    if query_lens:
                        self.query_lens = query_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.query_lens[seq_id] = 0

                    if context_lens:
                        self.context_lens = context_lens
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.context_lens[seq_id] = 0

                    if curr_sliding_window_blocks:
                        self.curr_sliding_window_blocks = curr_sliding_window_blocks
                    else:
                        for seq_id in range(len(self.seq_ids)):
                            self.curr_sliding_window_blocks[seq_id] = 0

                    if lora_index_mapping:
                        self.lora_index_mapping = lora_index_mapping
                    else:
                        self.lora_index_mapping.clear()

                    if lora_prompt_mapping:
                        self.lora_prompt_mapping = lora_prompt_mapping
                    else:
                        self.lora_prompt_mapping.clear()

                    if lora_requests:
                        self.lora_requests = lora_requests
                    else:
                        self.lora_requests.clear()

                    if prompt_adapter_index_mapping:
                        self.prompt_adapter_index_mapping = prompt_adapter_index_mapping
                    else:
                        self.prompt_adapter_index_mapping.clear()

                    if prompt_adapter_prompt_mapping:
                        self.prompt_adapter_prompt_mapping = prompt_adapter_prompt_mapping
                    else:
                        self.prompt_adapter_prompt_mapping.clear()

            else:
                self.input_tokens = input_tokens or []
                self.input_positions = input_positions or []
                self.seq_lens = seq_lens or []
                self.orig_seq_lens = orig_seq_lens or []
                self.query_lens = query_lens or []
                self.context_lens = context_lens or []
                self.curr_sliding_window_blocks = curr_sliding_window_blocks or []

                self.lora_index_mapping = lora_index_mapping or []
                self.lora_prompt_mapping = lora_prompt_mapping or []
                self.lora_requests = lora_requests or []

                self.prompt_adapter_index_mapping = prompt_adapter_index_mapping or []
                self.prompt_adapter_prompt_mapping = prompt_adapter_prompt_mapping or []

            self.prompt_adapter_request = prompt_adapter_request
            self.multi_modal_inputs = multi_modal_inputs
            self.prefix_cache_hit = prefix_cache_hit

            self.n_seqs = len(self.seq_ids)

            if not reinit:
                self.__post_init__()

        def __post_init__(self):
            self.n_seqs = len(self.seq_ids)

            self.input_tokens = [[] for _ in range(self.n_seqs)]
            self.input_positions = [[] for _ in range(self.n_seqs)]
            self.seq_lens = [0] * self.n_seqs
            self.orig_seq_lens = [0] * self.n_seqs
            self.query_lens = [0] * self.n_seqs
            self.context_lens = [0] * self.n_seqs
            self.curr_sliding_window_blocks = [0] * self.n_seqs

            self.lora_index_mapping = []
            self.lora_prompt_mapping = []

        def simple_reinit(self):
            self.input_tokens[0].clear()  # type: ignore
            self.input_positions[0].clear()  # type: ignore
            self.seq_lens[0] = 0  # type: ignore
            self.orig_seq_lens[0] = 0  # type: ignore
            self.query_lens[0] = 0  # type: ignore
            self.context_lens[0] = 0  # type: ignore
            self.curr_sliding_window_blocks[0] = 0  # type: ignore
            self.lora_index_mapping.clear()  # type: ignore
            self.lora_prompt_mapping.clear()  # type: ignore
            self.lora_requests.clear()  # type: ignore
            self.prompt_adapter_index_mapping.clear()  # type: ignore
            self.prompt_adapter_prompt_mapping.clear()  # type: ignore

    def __init__(self, runner: "NPUModelRunnerBase", finished_requests_ids: Optional[List[str]] = None):
        super().__init__()
        # Compute functions for each sequence in a sequence group.
        # WARNING: The order of the functions matters!
        self.per_seq_compute_fns = [
            self._compute_lens,
            self._compute_for_prefix_cache_hit,
            self._compute_for_sliding_window,
            self._compute_lora_input,
        ]
        # Compute functions for each sequence group.
        # WARNING: The order of the functions matters!
        self.per_seq_group_compute_fns = [
            self._compute_prompt_adapter_input,
            self._compute_multi_modal_input,
        ]

        self.runner = runner
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.scheduler_config = self.runner.scheduler_config
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.enable_lora = self.runner.lora_config is not None
        self.enable_prompt_adapter = self.runner.prompt_adapter_config is not None
        self.multi_modal_input_mapper = self.runner.multi_modal_input_mapper
        self.finished_requests_ids = finished_requests_ids
        self.decode_only = True

        # Intermediate data (data in CPU before going to NPU) for
        # the current sequence group.
        self.inter_data_list: List[ModelInputForNPUBuilder.InterDataForSeqGroup] = []

        # Attention metadata inputs.
        self.attn_metadata_builder = self.attn_backend.make_metadata_builder(weakref.proxy(self))

        # Engine/Model configurations.
        self.chunked_prefill_enabled = (
            self.scheduler_config is not None and self.scheduler_config.chunked_prefill_enabled
        )
        if self.sliding_window is not None:
            self.sliding_window_blocks = (self.sliding_window + self.block_size - 1) // self.block_size
            self.block_aligned_sliding_window = self.sliding_window_blocks * self.block_size

    def gen_inter_data_builder(self, num_seqs: int):
        return lambda: ModelInputForNPUBuilder.InterDataForSeqGroup(
            request_id="", seq_ids=[0] * num_seqs, is_prompt=True, block_tables=None, computed_block_nums=[]
        )

    def init_cached_inter_data(self, *args, **kwargs):
        assert len(args) == 0
        assert "seq_ids" in kwargs
        seq_ids = kwargs.get("seq_ids")
        num_seqs = len(seq_ids)

        # The inter-data cache is per model_runner
        inter_data_cache = self.runner.inter_data_cache
        if num_seqs not in inter_data_cache:
            inter_data_cache[num_seqs] = PyObjectCache(self.gen_inter_data_builder(num_seqs))

        obj = inter_data_cache[num_seqs].get_object()
        obj.__init__(*args, **kwargs)
        return obj

    def reset_cached_inter_data(self):
        for cache in self.runner.inter_data_cache.values():
            cache.reset()

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        """Add a sequence group to the builder."""
        seq_ids = seq_group_metadata.seq_data.keys()
        n_seqs = len(seq_ids)
        is_prompt = seq_group_metadata.is_prompt

        if is_prompt:
            assert n_seqs == 1
            self.decode_only = False

        inter_data = self.init_cached_inter_data(
            request_id=seq_group_metadata.request_id,
            seq_ids=seq_ids,
            is_prompt=is_prompt,
            block_tables=seq_group_metadata.block_tables,
            computed_block_nums=seq_group_metadata.computed_block_nums,
            reinit=True,
            reinit_use_defaults=True,
        )

        self.inter_data_list.append(inter_data)

        for seq_idx in range(n_seqs):
            for per_seq_fn in self.per_seq_compute_fns:
                per_seq_fn(inter_data, seq_idx, seq_group_metadata)
        for per_seq_group_fn in self.per_seq_group_compute_fns:
            per_seq_group_fn(inter_data, seq_group_metadata)

    def build(self) -> ModelInputForNPU:
        """Finalize the builder intermediate data and
        create on-device tensors.
        """
        # Combine and flatten intermediate data.
        input_tokens = []
        for inter_data in self.inter_data_list:
            for cur_input_tokens in inter_data.input_tokens:
                input_tokens.extend(cur_input_tokens)

        if not input_tokens:
            # This may happen when all prefill requests hit
            # prefix caching and there is no decode request.
            return self.model_input_cls()

        input_positions = []
        for inter_data in self.inter_data_list:
            for cur_input_positions in inter_data.input_positions:
                input_positions.extend(cur_input_positions)

        seq_lens = []
        max_decode_seq_len = 0
        for inter_data in self.inter_data_list:
            seq_lens.extend(inter_data.seq_lens)
            if not inter_data.is_prompt:
                max_decode_seq_len = max(max_decode_seq_len, max(inter_data.seq_lens))
        query_lens = []
        for inter_data in self.inter_data_list:
            query_lens.extend(inter_data.query_lens)

        # Mapping from request IDs to sequence IDs. Used for Jamba models
        # that manages the cache by itself.
        request_ids_to_seq_ids = {data.request_id: data.seq_ids for data in self.inter_data_list}

        batch_size = len(input_tokens)
        use_captured_graph = self._use_captured_graph(batch_size, max_decode_seq_len)

        # If cuda graph can be used, pad tensors accordingly.
        # See `capture_model` API for more details.
        # vLLM uses cuda graph only for decoding requests.
        cuda_graph_pad_size = -1
        if use_captured_graph:
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            cuda_graph_pad_size = graph_batch_size - batch_size
            batch_size = graph_batch_size

        # Tokens and positions.
        if cuda_graph_pad_size:
            input_tokens.extend(itertools.repeat(0, cuda_graph_pad_size))
            input_positions.extend(itertools.repeat(0, cuda_graph_pad_size))
        assert self.runner.device is not None
        input_tokens_tensor = async_tensor_h2d(input_tokens, torch.long, self.runner.device, self.runner.pin_memory)
        input_positions_tensor = async_tensor_h2d(
            input_positions, torch.long, self.runner.device, self.runner.pin_memory
        )

        # Sequence and query lengths.
        if cuda_graph_pad_size:
            seq_lens.extend(itertools.repeat(1, cuda_graph_pad_size))

        # Attention metadata.
        attn_metadata = self.attn_metadata_builder.build(seq_lens, query_lens, cuda_graph_pad_size, batch_size)

        # LoRA data.
        lora_requests = []
        lora_mapping = None
        if self.enable_lora:
            lora_requests = list(r if r else None for data in self.inter_data_list for r in data.lora_requests)
            lora_index_mapping = flatten_2d_lists(
                [flatten_2d_lists(inter_data.lora_index_mapping) for inter_data in self.inter_data_list]
            )
            if cuda_graph_pad_size:
                lora_index_mapping.extend(itertools.repeat(0, cuda_graph_pad_size))
            lora_prompt_mapping = flatten_2d_lists(
                [flatten_2d_lists(inter_data.lora_prompt_mapping) for inter_data in self.inter_data_list]
            )

            lora_mapping = LoRAMapping(
                **dict(
                    index_mapping=lora_index_mapping,
                    prompt_mapping=lora_prompt_mapping,
                    is_prefill=not self.decode_only,
                )
            )

        # Prompt adapter data.
        prompt_adapter_requests: Set[PromptAdapterRequest] = set()
        prompt_adapter_mapping = None
        if self.enable_prompt_adapter:
            prompt_adapter_requests = set(
                data.prompt_adapter_request for data in self.inter_data_list if data.prompt_adapter_request is not None
            )
            prompt_adapter_index_mapping = flatten_2d_lists(
                [inter_data.prompt_adapter_index_mapping for inter_data in self.inter_data_list]
            )
            if cuda_graph_pad_size:
                prompt_adapter_index_mapping.extend(itertools.repeat(0, cuda_graph_pad_size))
            prompt_adapter_prompt_mapping = flatten_2d_lists(
                [inter_data.prompt_adapter_prompt_mapping for inter_data in self.inter_data_list]
            )
            prompt_adapter_mapping = PromptAdapterMapping(
                prompt_adapter_index_mapping,
                prompt_adapter_prompt_mapping,
            )

        # Multi-modal data.
        multi_modal_inputs_list = [
            data.multi_modal_inputs for data in self.inter_data_list if data.multi_modal_inputs is not None
        ]
        multi_modal_kwargs = MultiModalInputs.batch(multi_modal_inputs_list)

        return self.model_input_cls(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            multi_modal_kwargs=multi_modal_kwargs,
            request_ids_to_seq_ids=request_ids_to_seq_ids,
            finished_requests_ids=self.finished_requests_ids,
            prompt_adapter_mapping=prompt_adapter_mapping,
            prompt_adapter_requests=prompt_adapter_requests,
        )

    def _compute_lens(self, inter_data: InterDataForSeqGroup, seq_idx: int, seq_group_metadata: SequenceGroupMetadata):
        """Compute context length, sequence length and tokens
        for the given sequence data.
        """
        seq_data = seq_group_metadata.seq_data[inter_data.seq_ids[seq_idx]]
        token_chunk_size = seq_group_metadata.token_chunk_size

        # Compute context length (the number of tokens that are
        # already computed) and sequence length (total number of tokens).
        seq_len = seq_data.get_len()
        if inter_data.is_prompt:
            context_len = seq_data.get_num_computed_tokens()
        else:
            # get_num_computed_tokens is incorrect for spec decoding.
            # So, we should have a special logic here.
            # TODO(sang): Fix it.
            context_len = seq_len - 1
        seq_len = min(seq_len, context_len + token_chunk_size)

        # Compute tokens.
        if inter_data.is_prompt:
            tokens = seq_data.get_token_ids()
            if context_len != 0 or seq_len < len(tokens):
                tokens = tokens[context_len:seq_len]
        else:
            # Optimization. get_token_ids requires the entire copy of
            # tokens.
            tokens = seq_data.get_last_token_id()

        inter_data.seq_lens[seq_idx] = seq_len
        inter_data.orig_seq_lens[seq_idx] = seq_len
        inter_data.context_lens[seq_idx] = context_len

        if isinstance(tokens, list):
            inter_data.input_tokens[seq_idx].extend(tokens)
        else:
            inter_data.input_tokens[seq_idx].append(tokens)

        if (seq_len - context_len) == 1:
            inter_data.input_positions[seq_idx].append(seq_len - 1)
        else:
            inter_data.input_positions[seq_idx].extend(range(context_len, seq_len))

        inter_data.query_lens[seq_idx] = seq_len - context_len if inter_data.is_prompt else 1

    def _compute_for_prefix_cache_hit(
        self, inter_data: InterDataForSeqGroup, seq_idx: int, seq_group_metadata: SequenceGroupMetadata
    ):
        """Check if hit prefix cache (i.e., some blocks are already computed).
        If hit, update input tokens and positions to only compute the
        remaining blocks.
        """
        computed_block_nums = inter_data.computed_block_nums

        # Note that prefix caching does not support sliding window.
        prefix_cache_hit = (
            computed_block_nums is not None
            and len(computed_block_nums) > 0
            and self.sliding_window is None
            and inter_data.is_prompt
        )
        inter_data.prefix_cache_hit = prefix_cache_hit

        if not prefix_cache_hit:
            return

        assert computed_block_nums is not None
        # The cache hit prompt tokens in this sequence. Note that
        # this may be larger than the sequence length if chunked
        # prefill is enabled.
        prefix_cache_len = len(computed_block_nums) * self.block_size
        # The number of so far computed prompt tokens in this sequence.
        context_len = inter_data.context_lens[seq_idx]
        # The total number of prompt tokens in this sequence.
        # When chunked prefill is enabled, this is the token number of
        # computed chunks + current chunk.
        seq_len = inter_data.seq_lens[seq_idx]
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            pass
        elif context_len < prefix_cache_len < seq_len:
            # Partial hit. Compute the missing part.
            uncomputed_start = prefix_cache_len - context_len
            inter_data.input_tokens[seq_idx] = inter_data.input_tokens[seq_idx][uncomputed_start:]
            inter_data.input_positions[seq_idx] = inter_data.input_positions[seq_idx][uncomputed_start:]
            context_len = prefix_cache_len

            inter_data.context_lens[seq_idx] = context_len
            inter_data.query_lens[seq_idx] = inter_data.seq_lens[seq_idx] - context_len
        elif seq_len <= prefix_cache_len:
            # Full hit. Only compute the last token to avoid
            # erroneous behavior. FIXME: Ideally we should directly
            # mark all tokens as computed in the scheduler and do not
            # schedule this sequence, so this case should not happen.
            inter_data.input_tokens[seq_idx] = inter_data.input_tokens[seq_idx][-1:]
            inter_data.input_positions[seq_idx] = inter_data.input_positions[seq_idx][-1:]
            inter_data.query_lens[seq_idx] = 1
            inter_data.context_lens[seq_idx] = inter_data.seq_lens[seq_idx] - 1

    def _compute_for_sliding_window(
        self, inter_data: InterDataForSeqGroup, seq_idx: int, seq_group_metadata: SequenceGroupMetadata
    ):
        """Update seq_len and curr_sliding_window_block for the given
        sequence data (only required by decoding) if sliding window is enabled.
        """
        curr_sliding_window_block = 0
        sliding_seq_len = inter_data.seq_lens[seq_idx]
        if not inter_data.is_prompt and self.sliding_window is not None:
            # TODO(sang): This is a hack to make sliding window work with
            # paged attn. We can remove it if we make paged attn kernel
            # to properly handle slinding window attn.
            curr_sliding_window_block = self.sliding_window_blocks
            if self.scheduler_config.use_v2_block_manager:
                # number of elements in last block
                suff_len = inter_data.seq_lens[seq_idx] % self.block_size
                sliding_seq_len = min(inter_data.seq_lens[seq_idx], self.block_aligned_sliding_window + suff_len)
                if suff_len > 0:
                    curr_sliding_window_block += 1
            else:
                sliding_seq_len = min(inter_data.seq_lens[seq_idx], self.sliding_window)

        inter_data.curr_sliding_window_blocks[seq_idx] = curr_sliding_window_block
        inter_data.seq_lens[seq_idx] = sliding_seq_len

    def _compute_lora_input(
        self, inter_data: InterDataForSeqGroup, seq_idx: int, seq_group_metadata: SequenceGroupMetadata
    ):
        """If LoRA is enabled, compute LoRA index and prompt mapping."""
        if not self.enable_lora:
            return

        lora_id = seq_group_metadata.lora_int_id
        if lora_id > 0:
            inter_data.lora_requests.append(seq_group_metadata.lora_request)
        query_len = inter_data.query_lens[seq_idx]
        inter_data.lora_index_mapping.append([lora_id] * query_len)
        inter_data.lora_prompt_mapping.append(
            [lora_id]
            * (
                query_len
                if seq_group_metadata.sampling_params and seq_group_metadata.sampling_params.prompt_logprobs is not None
                else 1
            )
        )

    def _compute_prompt_adapter_input(
        self, inter_data: InterDataForSeqGroup, seq_group_metadata: SequenceGroupMetadata
    ):
        """If prompt adapter is enabled, compute index and prompt mapping."""
        # Note that when is_prompt=True, we expect only one sequence
        # in the group.
        if not self.enable_prompt_adapter:
            return

        prompt_adapter_id = seq_group_metadata.prompt_adapter_id
        if prompt_adapter_id <= 0 or not inter_data.is_prompt:
            return

        # We expect only one sequence in the group when is_prompt=True.
        assert inter_data.n_seqs == 1
        query_len = inter_data.query_lens[0]
        inter_data.prompt_adapter_request = seq_group_metadata.prompt_adapter_request

        num_tokens = seq_group_metadata.prompt_adapter_num_virtual_tokens
        inter_data.prompt_adapter_index_mapping = [prompt_adapter_id] * num_tokens + [0] * (query_len - num_tokens)
        inter_data.prompt_adapter_prompt_mapping = [prompt_adapter_id] * (
            query_len
            if seq_group_metadata.sampling_params and seq_group_metadata.sampling_params.prompt_logprobs
            else 1
        )

    def _compute_multi_modal_input(self, inter_data: InterDataForSeqGroup, seq_group_metadata: SequenceGroupMetadata):
        """If multi-modal data is given, add it to the input."""
        mm_data = seq_group_metadata.multi_modal_data
        if not mm_data:
            return

        mm_kwargs = self.multi_modal_input_mapper(mm_data)
        inter_data.multi_modal_inputs = mm_kwargs

    def _use_captured_graph(self, batch_size: int, max_decode_seq_len: int) -> bool:
        return (
            self.decode_only
            and not self.runner.model_config.enforce_eager
            and batch_size <= self.runner.max_batchsize_to_capture
            and max_decode_seq_len <= self.runner.max_seq_len_to_capture
        )


class NPUModelRunnerBase(ModelRunnerBase[TModelInputForNPU]):
    """
    Helper class for shared methods between NPU model runners.
    """

    _model_input_cls: Type[TModelInputForNPU]
    _builder_cls: Type[ModelInputForNPUBuilder]

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        mindie_config: Dict[str, Any],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        return_hidden_states: bool = False,
        observability_config: Optional[ObservabilityConfig] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        self.prompt_adapter_config = prompt_adapter_config
        self.return_hidden_states = return_hidden_states
        self.observability_config = observability_config
        self.mindie_config = mindie_config

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture
        self.max_batchsize_to_capture = _get_max_graph_batch_size(self.scheduler_config.max_num_seqs)

        self.has_seqlen_agnostic = model_config.contains_seqlen_agnostic_layers(parallel_config)

        num_attn_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        self.attn_backend = (
            get_attn_backend(
                num_attn_heads,
                self.model_config.get_head_size(),
                self.model_config.get_num_kv_heads(self.parallel_config),
                self.model_config.get_sliding_window(),
                self.model_config.dtype,
                self.kv_cache_dtype,
                self.block_size,
            )
            if num_attn_heads
            else None
        )
        if self.attn_backend:
            self.attn_state = self.attn_backend.get_state_cls()(weakref.proxy(self))
        else:
            self.attn_state = CommonAttentionState(weakref.proxy(self))

        # Multi-modal data support
        self.input_registry = input_registry
        self.mm_registry = mm_registry
        self.multi_modal_input_mapper = mm_registry.create_input_mapper(model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None
        self.prompt_adapter_manager: LRUCacheWorkerPromptAdapterManager = None

        set_cpu_offload_max_bytes(int(self.cache_config.cpu_offload_gb * 1024**3))

        # Used to cache python objects
        self.inter_data_cache: Dict[int, PyObjectCache] = {}
        self.sampling_metadata_cache: SamplingMetadataCache = SamplingMetadataCache()

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:
            self.model = get_model(
                model_config=self.model_config,
                device_config=self.device_config,
                load_config=self.load_config,
                mindie_config=self.mindie_config,
                lora_config=self.lora_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
                cache_config=self.cache_config,
            )

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB", self.model_memory_usage / float(2**30))

        if self.lora_config:
            assert supports_lora(self.model), "Model does not support LoRA"
            assert not supports_multimodal(self.model), "To be tested: Multi-modal model with LoRA settings."

            logger.info("LoRA manager will be initialized in the MindIE backend.")

        # TODO: What is this prompt adapter?
        if self.prompt_adapter_config:
            self.prompt_adapter_manager = LRUCacheWorkerPromptAdapterManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.device,
                self.prompt_adapter_config,
            )
            self.model = self.prompt_adapter_manager.create_prompt_adapter_manager(self.model)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader.loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        from vllm.model_executor.model_loader.loader import TensorizerLoader

        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
        )

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_seq_len_to_capture + block_size - 1) // block_size

    def _prepare_model_input_tensors(
        self, seq_group_metadata_list: List[SequenceGroupMetadata], finished_requests_ids: Optional[List[str]] = None
    ) -> TModelInputForNPU:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        builder = self._builder_cls(weakref.proxy(self), finished_requests_ids)
        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)

        builder.reset_cached_inter_data()

        return builder.build()  # type: ignore

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request, rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)] for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # TODO: support MM models
        # Additional NPU memory may be needed for multi-modal encoding, which
        # needs to be accounted for when calculating the NPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for NPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = max_num_batched_tokens // max_num_seqs + (group_id < max_num_batched_tokens % max_num_seqs)
            batch_size += seq_len

            seq_data, dummy_multi_modal_data = self.input_registry.dummy_data_for_profiling(
                self.model_config, seq_len, self.mm_registry
            )

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id] if dummy_lora_requests_per_seq else None,
                multi_modal_data=dummy_multi_modal_data,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.npu.synchronize()
        return

    def remove_all_loras(self): ...

    def set_active_loras(self, lora_requests: Set[LoRARequest], lora_mapping: LoRAMapping) -> None: ...

    def add_lora(self, lora_request: LoRARequest) -> bool: ...

    def remove_lora(self, lora_id: int) -> bool: ...

    def pin_lora(self, lora_id: int) -> bool: ...

    def list_loras(self) -> Set[int]: ...

    def remove_all_prompt_adapters(self): ...

    def set_active_prompt_adapters(
        self, prompt_adapter_requests: Set[PromptAdapterRequest], prompt_adapter_mapping: PromptAdapterMapping
    ) -> None: ...

    def add_prompt_adapter(self, prompt_adapter_request: PromptAdapterRequest) -> bool: ...

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool: ...

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool: ...

    def list_prompt_adapters(self) -> Set[int]: ...


class ModelRunner(NPUModelRunnerBase[ModelInputForNPUWithSamplingMetadata]):
    """
    NPU model runner with sampling step.
    """

    _model_input_cls: Type[ModelInputForNPUWithSamplingMetadata] = ModelInputForNPUWithSamplingMetadata
    _builder_cls: Type[ModelInputForNPUBuilder] = ModelInputForNPUBuilder

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForNPUWithSamplingMetadata:
        model_input = ModelInputForNPUWithSamplingMetadata.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForNPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        model_input = self._prepare_model_input_tensors(seq_group_metadata_list, finished_requests_ids)
        if get_pp_group().is_last_rank:
            # Sampling metadata is only required for the final pp group
            generators = self.get_generators(finished_requests_ids)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list,
                model_input.seq_lens,
                model_input.query_lens,
                self.device,
                self.pin_memory,
                generators,
                self.sampling_metadata_cache,
            )
        else:
            sampling_metadata = None
        is_prompt = seq_group_metadata_list[0].is_prompt if seq_group_metadata_list else None
        return dataclasses.replace(
            model_input, sampling_metadata=sampling_metadata, is_prompt=is_prompt, virtual_engine=virtual_engine
        )

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForNPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        self.attn_state.begin_forward(model_input)

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][graph_batch_size]
        else:
            model_executable = self.model

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = (
            {
                "finished_requests_ids": model_input.finished_requests_ids,
                "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
            }
            if self.has_seqlen_agnostic
            else {}
        )
        if self.observability_config is not None and self.observability_config.collect_model_forward_time:
            model_forward_start = torch.npu.streams.Event(enable_timing=True)
            model_forward_end = torch.npu.streams.Event(enable_timing=True)
            model_forward_start.record()

        hidden_or_intermediate_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            lora_requests=model_input.lora_requests,
            **MultiModalInputs.as_kwargs(multi_modal_kwargs, device=self.device),
            **seqlen_agnostic_kwargs,
        )

        if self.observability_config is not None and self.observability_config.collect_model_forward_time:
            model_forward_end.record()

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            if (
                self.is_driver_worker
                and hidden_or_intermediate_states is not None
                and isinstance(hidden_or_intermediate_states, IntermediateTensors)
                and self.observability_config is not None
                and self.observability_config.collect_model_forward_time
            ):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)
                    ).item()
                hidden_or_intermediate_states.tensors["model_forward_time"] = torch.tensor(
                    model_forward_time + orig_model_forward_time
                )
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states, model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        if (
            self.observability_config is not None
            and self.observability_config.collect_model_forward_time
            and output is not None
        ):
            model_forward_end.synchronize()
            model_forward_time = model_forward_start.elapsed_time(model_forward_end)
            orig_model_forward_time = 0.0
            if intermediate_tensors is not None:
                orig_model_forward_time = intermediate_tensors.tensors.get(
                    "model_forward_time", torch.tensor(0.0)
                ).item()
            # If there are multiple workers, we are still tracking the latency
            # from the start time of the driver worker to the end time of the
            # driver worker. The model forward time will then end up covering
            # the communication time as well.
            output.model_forward_time = orig_model_forward_time + model_forward_time

        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(0, indices)
                output.prefill_hidden_states = hidden_or_intermediate_states
            elif decode_meta.use_cuda_graph:
                hidden_states = hidden_or_intermediate_states[: len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]


def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return (batch_size + _BATCH_SIZE_ALIGNMENT - 1) // _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT


def _get_max_graph_batch_size(max_num_seqs: int) -> int:
    """
    max_num_seqs: Maximum number of sequences in a batch.
    _BATCH_SIZES_TO_CAPTURE: all the sizes that we want to capture.

    pad the max_num_seqs if necessary by calling _get_graph_batch_size,
    which will deal with some edge cases like 1, 2, 4.

    if the padded size is in _BATCH_SIZES_TO_CAPTURE, return the padded size.
    if not, it means the padded size is larger than the largest size in
    _BATCH_SIZES_TO_CAPTURE, return the largest size in _BATCH_SIZES_TO_CAPTURE.
    """
    padded_size = _get_graph_batch_size(max_num_seqs)
    if padded_size in _BATCH_SIZES_TO_CAPTURE:
        return padded_size
    assert padded_size > _BATCH_SIZES_TO_CAPTURE[-1]
    return _BATCH_SIZES_TO_CAPTURE[-1]


def seq_output_builder():
    return SequenceOutput(0, 0, {0: Logprob(logprob=float("inf"), rank=None, decoded_token=None)})


def completion_seq_group_output_builder():
    return CompletionSequenceGroupOutput([], None)


class PythonizationCache:

    def __init__(self):
        self.cached_seq_output = PyObjectCache(seq_output_builder)
        self.cached_completion_seq_group_output = PyObjectCache(completion_seq_group_output_builder)

    def reset(self):
        self.cached_seq_output.reset()
        self.cached_completion_seq_group_output.reset()


@dataclass
class ModelOutput:
    sampler_output: SamplerOutput
    sampler_output_ready_event: torch.npu.streams.Event
    sampled_token_ids: Optional[torch.Tensor] = None
    pythonized: bool = False
    # On-device tensor containing the logprobs of each token.
    logprobs: Optional["torch.Tensor"] = None
    pythonization_cache: Optional[PythonizationCache] = None

    def pythonize(
        self,
        input_metadata: "StatefulModelInput",
        copy_stream: torch.npu.streams.Stream,
        pinned_sampled_token_buffer: torch.Tensor,
    ) -> None:
        """Pythonize the output. Blocking."""
        if not self.pythonized:
            self._pythonize_sampler_output(input_metadata, copy_stream, pinned_sampled_token_buffer, True)
            self.pythonized = True

    def maybe_pythonize(
        self,
        input_metadata: "StatefulModelInput",
        copy_stream: torch.npu.streams.Stream,
        pinned_sampled_token_buffer: torch.Tensor,
    ) -> None:
        """Pythonize the output if ready, else return None. Non-blocking."""
        if not self.pythonized:
            self.pythonized = self._pythonize_sampler_output(
                input_metadata, copy_stream, pinned_sampled_token_buffer, False
            )

    def _pythonize_sampler_output(
        self,
        input_metadata: "StatefulModelInput",
        copy_stream: torch.npu.streams.Stream,
        pinned_sampled_token_buffer: torch.Tensor,
        blocking: bool,
    ) -> bool:
        """
        If blocking is set, will block until the forward pass for the output is
        ready and pythonize the output. Upon completing Pythonization, erases
        self.logprobs (note that a non-blocking call that is performed when
        the sampler output is not yet ready, will not erase self.logprobs.)
        """
        assert self.sampled_token_ids is not None
        if not blocking and not self.sampler_output_ready_event.query():
            return False

        if blocking:
            self.sampler_output_ready_event.synchronize()
        with torch.npu.utils.stream(copy_stream):
            _pythonize_sampler_output(
                input_metadata,
                self.sampler_output,
                pinned_sampled_token_buffer,
                self.sampled_token_ids,
                self.logprobs,
                self.pythonization_cache,
            )
        self.logprobs = None
        return True


@dataclass(frozen=False)
class StatefulModelInput(BroadcastableModelInput):
    # actual frozen model input dataclass passed to _base_model_runner
    frozen_model_input: Optional[ModelInputForNPUWithSamplingMetadata] = None

    # list of model outputs for each step, may not be all pythonized
    cached_outputs: List[ModelOutput] = field(default_factory=list)

    # used to pass sampled token ids from the last step to the current step for
    # TP workers. Used to append to end of outputs and used by advance_step
    last_sampled_token_ids: Optional[torch.Tensor] = None
    current_step: int = 0
    is_multi_step: bool = True
    is_last_step: bool = False
    is_first_multi_step: bool = False
    # ping-pong data structures for multi-step to wait on the previous step
    step_npu_events: List[torch.npu.streams.Event] = field(
        default_factory=lambda: [torch.npu.streams.Event(blocking=True)] * 2
    )
    num_seqs: int = -1
    num_queries: int = -1

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "StatefulModelInput":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(attn_backend, tensor_dict)
        tensor_dict = _init_frozen_model_input_from_tensor_dict(ModelInputForNPUWithSamplingMetadata, tensor_dict)

        return cls(**tensor_dict)

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        assert self.frozen_model_input is not None
        tensor_dict = self.frozen_model_input.as_broadcastable_tensor_dict()
        new_tensor_dict = {
            "last_sampled_token_ids": self.last_sampled_token_ids,
            "current_step": self.current_step,
            "is_multi_step": self.is_multi_step,
            "is_last_step": self.is_last_step,
            "is_first_multi_step": self.is_first_multi_step,
            "num_seqs": self.num_seqs,
            "num_queries": self.num_queries,
        }
        tensor_dict.update(new_tensor_dict)
        return tensor_dict

    def record_step_event(self, current_stream: torch.npu.streams.Stream):
        # record the event for the current step so that the next step can sync
        # on it. We modulo by 2 to keep the events in a circular buffer and
        # support any attn backends that may be supported in the future. ie
        # Flashinfer would want two DecodeWrappers to overlap the CPU and NPU.
        self.step_npu_events[self.current_step & 1] = torch.npu.streams.Event(blocking=True)
        self.step_npu_events[self.current_step & 1].record(current_stream)

    def wait_previous_step(self):
        # These cuda events are an explicit synchronization to ensure that
        # advance_step() (for other attn backends that may be supported in the
        # future) do not clobber any data structures that is also used by any
        # enqueued forwards steps. For distributed case, only a single event is
        # needed, but for single NPU case, since we can let the CPU run much
        # further ahead, two events allow us to overlap the advance_step with
        # the previous forward (ie using two DecodeWrappers for flashinfer
        # backend)
        self.step_npu_events[(self.current_step + 1) & 1].wait()

    def add_sampler_output(self, sampler_output: SamplerOutput, sampled_token_ids: Optional[torch.Tensor] = None):
        self.cached_outputs.append(
            ModelOutput(
                sampler_output=sampler_output,
                sampler_output_ready_event=None,
                sampled_token_ids=sampled_token_ids,
                pythonized=False,
            )
        )


class MultiStepNPUModelRunner(NPUModelRunnerBase[StatefulModelInput]):

    def __init__(self, base_model_runner: NPUModelRunnerBase, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # uses the base model runner to execute the model and wraps it with
        # multi-step logic
        self._base_model_runner: NPUModelRunnerBase = base_model_runner

        self.is_multi_step = self.scheduler_config.is_multi_step
        self.pinned_sampled_token_ids: Optional[torch.Tensor] = None

        self.pythonization_cache = PythonizationCache()

    @property
    def vocab_size(self) -> int:
        return self._base_model_runner.vocab_size

    def make_model_input_from_broadcasted_tensor_dict(self, tensor_dict: Dict[str, Any]) -> StatefulModelInput:
        model_input = StatefulModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> StatefulModelInput:
        frozen_model_input = self._base_model_runner.prepare_model_input(
            seq_group_metadata_list, virtual_engine, finished_requests_ids
        )

        model_input = StatefulModelInput(
            frozen_model_input=frozen_model_input,
            num_seqs=len(frozen_model_input.seq_lens),
            num_queries=len(frozen_model_input.query_lens),
        )
        return model_input

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: StatefulModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        """
        Execute the model for a single step and update multi-step
        metadata
        """
        assert num_steps == 1, "MultiStepModelRunner only supports num_steps=1"
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None

        # path for warm up runs
        if not model_input.is_multi_step:
            return self._base_model_runner.execute_model(frozen_model_input, kv_caches, intermediate_tensors, num_steps)

        # make sure we skip the sampler on the lask rank and only pythonize
        # if CPU is ahead.
        if self.is_driver_worker and get_pp_group().is_last_rank:
            if self.pinned_sampled_token_ids is None:
                self.pinned_sampled_token_ids = torch.zeros(
                    (self.scheduler_config.max_num_seqs, 1), dtype=torch.long, device="cpu", pin_memory=True
                )

            self._base_model_runner.model.sampler.include_gpu_probs_tensor = True
            if frozen_model_input.sampling_metadata:
                frozen_model_input.sampling_metadata.skip_sampler_cpu_output = True

        # some pre-execute model logic for multi-step:
        #   - if it's the first step, we need to reset the sampling tensors
        #   - if it's not the first step, we need to advance the step using the
        #   appended sampler output from last iteration
        #   - also maybe pythonize if CPU is ahead of NPU

        current_stream = torch.npu.utils.current_stream()
        if not model_input.is_first_multi_step:
            # Explicitly block on the previous step's forward to make sure we
            # don't clobber any NPU tensors still in use.
            # This is not needed for flashattn backend, but for other attn
            # backends such as flashinfer that performs extra CPU operations on
            # input metadata we may need to synchronize any CPU operations that
            # might clobber enqueued forwards. (prevents CPU from running too
            # far ahead if needed)
            model_input.wait_previous_step()
            model_input = self._advance_step(model_input, model_input.cached_outputs[-1].sampler_output)

        output_proc_callback = None
        if frozen_model_input.async_callback is not None:
            output_proc_callback = frozen_model_input.async_callback
            assert output_proc_callback is not None
            async_callback = functools.partial(
                self._async_process_outputs, model_input=model_input, output_proc_callback=output_proc_callback
            )

            frozen_model_input = dataclasses.replace(  # type: ignore
                model_input.frozen_model_input, async_callback=async_callback
            )
            assert frozen_model_input is not None

        # Execute the model
        output = self._base_model_runner.execute_model(frozen_model_input, kv_caches, intermediate_tensors, num_steps=1)

        # record the event for the current step so that the next step can sync
        model_input.record_step_event(current_stream)

        if get_pp_group().is_last_rank and self.is_driver_worker:
            assert len(output) == 1, "MultiStepModelRunner requires single-step base_models"

            # event for the pythonization so that we only pythonize if the
            # tensors are ready. May be able to be combined with the step event
            output_ready_event = torch.npu.streams.Event()
            output_ready_event.record(current_stream)
            if self.parallel_config.pipeline_parallel_size > 1:
                output[0].sampled_token_ids_cpu = output[0].sampled_token_ids.cpu()
            model_input.cached_outputs.append(
                ModelOutput(
                    output[0],
                    output_ready_event,
                    output[0].sampled_token_ids,
                    False,
                    output[0].logprobs,
                    self.pythonization_cache,
                )
            )

            # These NPU tensors are not required by multi-step;
            # erase them to ensure they are not pythonized or
            # transferred to CPU
            output[0].sampled_token_ids = None
            output[0].sampled_token_probs = None
            output[0].logprobs = None

            # Pythonize the output if CPU is ahead and the previous step is
            # ready.
            if frozen_model_input.async_callback is None:
                for model_output in model_input.cached_outputs:
                    model_output.maybe_pythonize(model_input, self._copy_stream, self.pinned_sampled_token_ids)

        model_input.current_step += 1

        if not get_pp_group().is_last_rank:
            # Should be IntermediateTensors
            assert isinstance(output, IntermediateTensors)
            return output
        if not self.is_driver_worker:
            return []

        # Pythonize the output and block if needed since it is the last step
        if model_input.is_last_step:
            outputs = self._final_process_outputs(model_input, output_proc_callback)
            self.pythonization_cache.reset()
            return outputs

        # should be [SamplerOutput]
        return output

    def load_model(self) -> None:
        return self._base_model_runner.load_model()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        return self._base_model_runner.save_sharded_state(path, pattern, max_size)

    def save_tensorized_model(self, tensorizer_config: TensorizerConfig) -> None:
        return self._base_model_runner.save_tensorized_model(tensorizer_config)

    def profile_run(self) -> None:
        return self._base_model_runner.profile_run()

    def remove_all_loras(self):
        return self._base_model_runner.remove_all_loras()

    def capture_model(self, kv_caches: List[List]) -> None:
        return self._base_model_runner.capture_model(kv_caches)

    @functools.cached_property
    def _copy_stream(self):
        # used to copy tensors from NPU to CPU asynchronously
        return torch.npu.streams.Stream()

    def _async_process_outputs(self, model_input: StatefulModelInput, output_proc_callback: Callable):
        # Proceed with pythonization and output_proc in order.
        # Stop on the first one that fails to pythonize
        output_proc_callback()

        cont = True
        for model_output in model_input.cached_outputs:
            if not model_output.pythonized:
                model_output.maybe_pythonize(model_input, self._copy_stream, self.pinned_sampled_token_ids)
                if model_output.pythonized:
                    ctx = output_proc_callback.keywords["ctx"]
                    ctx.append_output(
                        outputs=[model_output.sampler_output],
                        seq_group_metadata_list=ctx.seq_group_metadata_list,
                        scheduler_outputs=ctx.scheduler_outputs,
                        is_async=False,
                        is_last_step=False,
                    )

                    output_proc_callback()
                else:
                    cont = False

            if not cont:
                break

    def _final_process_outputs(self, model_input: StatefulModelInput, output_proc_callback: Optional[Callable]):
        assert model_input.frozen_model_input is not None

        has_async_callback = output_proc_callback is not None

        outputs = []
        for output_id in range(len(model_input.cached_outputs)):
            output = model_input.cached_outputs[output_id]
            is_last_step = output_id == len(model_input.cached_outputs) - 1

            # For non-async case:
            #   -- We simply add the outputs
            # For async case:
            #   -- Invoke callback, pythonize, add to callback queue and repeat
            #   -- For last output, just add to callback queue
            if has_async_callback:
                assert output_proc_callback is not None

                # Invoke callback before pythonize (to overlap with NPU)
                output_proc_callback()

                # Pythonize
                if not output.pythonized:
                    output.pythonize(model_input, self._copy_stream, self.pinned_sampled_token_ids)

                    # For non last step, add to callback queue to chain
                    # callbacks=>pythonize pairs (for NPU overlap)
                    if not is_last_step:
                        ctx = output_proc_callback.keywords[  # type: ignore
                            "ctx"
                        ]  # type: ignore
                        ctx.append_output(
                            outputs=[output.sampler_output],
                            seq_group_metadata_list=ctx.seq_group_metadata_list,
                            scheduler_outputs=ctx.scheduler_outputs,
                            is_async=False,
                            is_last_step=False,
                        )
                    else:
                        outputs.append(output.sampler_output)
            else:
                output.pythonize(model_input, self._copy_stream, self.pinned_sampled_token_ids)
                outputs.append(output.sampler_output)

        return outputs

    def _update_sampling_metadata(self, sampling_metadata, num_seqs, num_queries):

        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_queries
        assert sampling_metadata.selected_token_indices.shape == (num_queries,)
        # assert sampling_metadata.categorized_sample_indices == TODO: Add if needed # noqa: E501

        # Verify that all sequences are decodes
        for i in range(num_queries):
            seq_group = sampling_metadata.seq_groups[i]

            assert seq_group.is_prompt is False  # No prompt
            assert seq_group.prompt_logprob_indices == []  # No prompt
            assert seq_group.sample_indices == [i]  # Simple
            assert seq_group.seq_len is None  # Decode
            assert seq_group.query_len is None  # Decode

    def _advance_step(self, model_input: StatefulModelInput, out: SamplerOutput) -> StatefulModelInput:
        if self.attn_backend.get_name() not in MULTI_STEP_ATTENTION_BACKENDS:
            raise ValueError(
                f"Multi-step not supported for attention backend: "
                f"{self.attn_backend.get_name()}. Set VLLM_ATTENTION_BACKEND "
                f"to a value from {MULTI_STEP_ATTENTION_BACKENDS}."
            )

        sampled_token_ids = model_input.cached_outputs[-1].sampled_token_ids
        num_seqs = model_input.num_seqs
        num_queries = model_input.num_queries
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None
        attn_metadata = frozen_model_input.attn_metadata
        assert attn_metadata is not None

        attn_metadata.advance_step(
            frozen_model_input,
            sampled_token_ids,
            self.block_size,
            num_seqs,
            num_queries,
        )

        return model_input


DeferredLogprobsReturnType = Tuple[Optional[List[Optional[PromptLogprobs]]], Optional[List[SampleLogprobs]]]


def deferred_pythonize_logprobs(
    output: SamplerOutput,
    sampling_metadata: SamplingMetadata,
    logprobs_tensor: Optional[torch.Tensor],
) -> DeferredLogprobsReturnType:
    """Perform deferred logprob Pythonization.

    1. Pythonize NPU-side sampler result tensors into CPU-side sampler result.
    2. Pythonize NPU-side logprobs tensor into CPU-side logprobs lists,
       utilizing  the Pythonized sampler result computed in step 1.

    These deferred computations are not required for single-step scheduling
    or the `profile_run()` phase of multi-step scheduling.

    Args:
        output: sampler output (under deferred Pythonization)
        sampling_metadata

    Returns:
        prompt_logprobs (CPU), sample_logprobs (CPU)
    """

    # - Deferred pythonization of sample result
    sampler_result = get_pythonized_sample_results(output.deferred_sample_results_args)

    # - Erase the NPU-side deferred sample_result
    #   computation args to ensure it is never
    #   pythonized or transferred to CPU
    output.deferred_sample_results_args = None

    # - Deferred pythonization of logprobs
    (
        prompt_logprobs,
        sample_logprobs,
    ) = get_logprobs(logprobs_tensor, sampling_metadata, sampler_result)
    assert len(prompt_logprobs) == len(sampling_metadata.seq_groups)
    assert len(sample_logprobs) == len(sampling_metadata.seq_groups)

    return prompt_logprobs, sample_logprobs


def _pythonize_sampler_output(
    model_input: StatefulModelInput,
    output: SamplerOutput,
    pinned_sampled_token_buffer: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    logprobs_tensor: Optional[torch.Tensor],
    cache: Optional[PythonizationCache],
) -> None:
    assert model_input.frozen_model_input is not None

    frozen_model_input = model_input.frozen_model_input
    assert frozen_model_input.sampling_metadata is not None
    sampling_metadata = frozen_model_input.sampling_metadata
    # samples generation should have been skipped
    # assert not output.outputs

    pinned_buffer = pinned_sampled_token_buffer[: model_input.num_queries]

    # We guarantee output tensors are ready, so it is safe to
    # pythonize the sampler output & obtain CPU-side logprobs.
    #
    # However we should check whether logprobs pythonization may
    # be skipped entirely, i.e. because no logprobs were requested
    # or pythonization was not deferred. To that end,
    #
    # * `prompt_logprobs_are_requested_for_prefill` signals that
    #   there are *any* prefill-phase requests which specify that
    #   prompt logprobs should be returned.
    #
    # * `any_logprobs_are_requested` signals that there are any
    #   requests which (1) specify that sample logprobs should be
    #   returned, or (2) are in the prefill phase AND specify that
    #   prompt logprobs should be returned.
    #
    # Later on, these flags cause adjustments to the pythonization
    # process to accommodate logprobs.

    seq_groups = sampling_metadata.seq_groups
    prompt_logprobs_are_requested_for_prefill = any(
        [sg.sampling_params.prompt_logprobs is not None and sg.is_prompt for sg in seq_groups]
    )
    any_logprobs_are_requested = prompt_logprobs_are_requested_for_prefill or any(
        [sg.sampling_params.logprobs is not None for sg in seq_groups]
    )

    if prompt_logprobs_are_requested_for_prefill:
        # CPU NPU sync, after gathering *only* sampled tokens (since
        # requesting prompt logprobs leads `sampled_token_ids` to
        # include prompt token ids in addition to sampled token ids.)
        sample_idx_tensor = torch.tensor([sdx for sg in seq_groups for sdx in sg.sample_indices])
        pinned_buffer = pinned_buffer.copy_(sampled_token_ids[sample_idx_tensor, :], non_blocking=False)
    else:
        # CPU NPU sync
        pinned_buffer = pinned_buffer.copy_(sampled_token_ids, non_blocking=False)

    # this will not block as the tensors are already on CPU
    samples_list = pinned_buffer.tolist()

    skip_sampler_cpu_output = frozen_model_input.sampling_metadata.skip_sampler_cpu_output

    # *Don't* skip logprobs pythonization *if*:
    # * Any requests require logprobs to be returned in this
    # iteration AND
    # * These requests are being scheduled in a fashion which
    # defers pythonization (i.e. multi-step scheduling.)
    do_pythonize_logprobs = skip_sampler_cpu_output and any_logprobs_are_requested
    (
        prompt_logprobs,
        sample_logprobs,
    ) = (
        deferred_pythonize_logprobs(output, sampling_metadata, logprobs_tensor)
        if do_pythonize_logprobs
        else (None, None)
    )

    for sgdx, (seq_group, sample_result) in enumerate(zip(seq_groups, samples_list)):
        if seq_group.sampling_params.logits_processors:
            assert (
                len(seq_group.sampling_params.logits_processors) == 0
            ), "Logits Processors are not supported in multi-step decoding"

        if do_pythonize_logprobs:
            assert prompt_logprobs is not None
            assert sample_logprobs is not None

            (
                group_prompt_logprobs,
                group_sample_logprobs,
            ) = (  # Utilize deferred pythonization results
                prompt_logprobs[sgdx],
                sample_logprobs[sgdx],
            )
        elif any_logprobs_are_requested:
            (
                group_prompt_logprobs,
                group_sample_logprobs,
            ) = (
                # profile_run: use already-computed logprobs
                output.outputs[sgdx].prompt_logprobs,
                [sample.logprobs for sample in output.outputs[sgdx].samples],
            )

        seq_ids = seq_group.seq_ids
        next_token_ids = sample_result
        parent_ids = [0]

        if cache is not None:
            completion_seq_group_output: CompletionSequenceGroupOutput = (
                cache.cached_completion_seq_group_output.get_object()
            )
            completion_seq_group_output.samples.clear()
            seq_outputs: List[SequenceOutput] = completion_seq_group_output.samples
        else:
            seq_outputs = []

        for tdx, (parent_id, next_token_id) in enumerate(zip(parent_ids, next_token_ids)):
            if cache is not None:
                seq_output: SequenceOutput = cache.cached_seq_output.get_object()
                seq_output.parent_seq_id = seq_ids[parent_id]
                seq_output.output_token = next_token_id

                if any_logprobs_are_requested:
                    seq_output.logprobs = group_sample_logprobs[tdx]
                else:
                    logprobs = next(iter(seq_output.logprobs.values()))
                    seq_output.logprobs.clear()

                    logprobs.logprob = float("inf")
                    logprobs.rank = None
                    logprobs.decoded_token = None

                    seq_output.logprobs[next_token_id] = logprobs

                seq_outputs.append(seq_output)

            else:
                seq_outputs.append(
                    SequenceOutput(
                        seq_ids[parent_id],
                        next_token_id,
                        (
                            group_sample_logprobs[tdx]
                            if any_logprobs_are_requested
                            else {next_token_id: Logprob(logprob=float("inf"), rank=None, decoded_token=None)}
                        ),
                    )
                )
        if cache is not None:
            completion_seq_group_output.prompt_logprobs = group_prompt_logprobs if any_logprobs_are_requested else None
            output.outputs.append(completion_seq_group_output)
        else:
            output.outputs.append(
                CompletionSequenceGroupOutput(
                    seq_outputs, (group_prompt_logprobs if any_logprobs_are_requested else None)
                )
            )

    assert len(output.outputs) > 0
