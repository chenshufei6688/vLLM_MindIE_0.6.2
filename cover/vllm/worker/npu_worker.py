# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# Part of codes in this file was copied from project [vLLM Team][vllm]

"""An NPU worker class."""
import dataclasses
from dataclasses import dataclass
import gc
from typing import List, Optional, Set, Tuple, Dict

import torch
import torch.distributed
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
    SpeculativeConfig,
)
from vllm.sequence import ExecuteModelRequest
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.npu_model_runner import ModelRunner
from vllm.worker.worker_base import LocalOrDistributedWorkerBase, WorkerInput, extract_previous_hidden_states
from vllm.worker.model_runner_base import BroadcastableModelInput, ModelRunnerInputBase
from vllm.worker.npu_model_runner import MultiStepNPUModelRunner, StatefulModelInput
from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.distributed.npu_utils import ascend_broadcast_data_dict


class NPUWorker(LocalOrDistributedWorkerBase):
    """A worker class that executes the model on a group of Ascend NPUs."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        is_driver_worker: bool = False,
        observability_config: Optional[ObservabilityConfig] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        if parallel_config and is_driver_worker:
            assert (
                rank % parallel_config.tensor_parallel_size == 0
            ), "Driver worker should be rank 0 of tensor parallel group."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        mindie_config = {
            "backend_type": "atb",
            "model_id": model_config.model,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": parallel_config.world_size,
            "npu_device_id": local_rank,
            "trust_remote_code": model_config.trust_remote_code,
            "inference_mode": (
                2 if scheduler_config.chunked_prefill_enabled or cache_config.enable_prefix_caching else 0
            ),
        }
        self.model_runner = ModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config,
            lora_config,
            mindie_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
        )
        # Uninitialized cache engine. Will be initialized by
        # self.initialize_cache().
        self.cache_engine: List[CacheEngine]
        self.npu_cache: List[torch.Tensor]

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.npu_cache

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    def init_device(self) -> None:
        self.device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(self.device)
        gc.collect()
        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.parallel_config, self.rank, self.distributed_init_method, self.local_rank
        )
        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of NPU and CPU cache blocks that can be allocated.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
        block_size = self.cache_config.block_size
        dummy_block_size = 128
        dummy_num_blocks = dummy_block_size // block_size * self.model_runner.model.dummy_block_num

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.npu.synchronize()
        peak_memory = torch.npu.max_memory_allocated()

        total_gpu_memory = torch.npu.get_device_properties(self.rank).total_memory
        cache_block_size = CacheEngine.get_cache_block_size(self.cache_config, self.model_config, self.parallel_config)

        num_gpu_blocks = (
            int((total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_memory) // cache_block_size)
            + dummy_num_blocks
        )
        num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.npu.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        raise_if_cache_size_invalid(num_gpu_blocks, self.cache_config.block_size, self.model_config.max_model_len)
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _get_worker_input_from_broadcast(
        self
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
            str, torch.Tensor]]]:
        """ Get the worker input from the broadcasted tensor dict. """
        assert self.do_metadata_broadcast
        assert not self.is_driver_worker
        broadcast_data = ascend_broadcast_data_dict(src=0)
        if not broadcast_data:
            return None

        worker_input = WorkerInput.from_broadcasted_tensor_dict(broadcast_data)
        model_input = (
            self.model_runner.make_model_input_from_broadcasted_tensor_dict(
                broadcast_data))

        kwargs = extract_previous_hidden_states(broadcast_data)

        return model_input, worker_input, kwargs

    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]:
        """ Get the driver input and broadcast it to other workers.  """
        assert self.is_driver_worker

        worker_input: WorkerInput = self.prepare_worker_input(
            execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = (
            self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.virtual_engine,
                execute_model_req.finished_requests_ids))

        kwargs = extract_previous_hidden_states(execute_model_req)

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_data.update(kwargs)
            ascend_broadcast_data_dict(broadcast_data, src=0)
        if execute_model_req.async_callback:
            model_input = dataclasses.replace(  # type: ignore
                model_input,
                async_callback=execute_model_req.async_callback)

        return model_input, worker_input, kwargs

    @torch.inference_mode()
    def prepare_worker_input(self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_steps = execute_model_req.num_steps
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in, device="cpu", dtype=torch.int64).view(
            -1, 2
        )
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out, device="cpu", dtype=torch.int64).view(
            -1, 2
        )
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy, device=self.device, dtype=torch.int64).view(
            -1, 2
        )

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
            num_steps=num_steps,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if worker_input.blocks_to_swap_in is not None and worker_input.blocks_to_swap_in.numel() > 0:
            self.cache_engine[virtual_engine].swap_in(worker_input.blocks_to_swap_in)
        if worker_input.blocks_to_swap_out is not None and worker_input.blocks_to_swap_out.numel() > 0:
            self.cache_engine[virtual_engine].swap_out(worker_input.blocks_to_swap_out)
        if worker_input.blocks_to_copy is not None and worker_input.blocks_to_copy.numel() > 0:
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return self.model_runner.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.remove_lora(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.pin_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        return self.model_runner.list_prompt_adapters()

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes."""
        return CacheEngine.get_cache_block_size(self.cache_config, self.model_config, self.parallel_config)

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            CacheEngine(self.cache_config, self.model_config, self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.npu_cache = [self.cache_engine[ve].gpu_cache for ve in range(self.parallel_config.pipeline_parallel_size)]

    def _warm_up_model(self) -> None:
        pass


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank, "hccl")

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size)


def raise_if_cache_size_invalid(num_gpu_blocks, block_size, max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when "
            "initializing the engine."
        )
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine."
        )


@dataclass
class MultiStepState:
    worker_input: WorkerInput
    model_input: StatefulModelInput


class MultiStepNPUWorker(NPUWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_model_runner = self.model_runner
        # for multi-step model, wrap the model runner with MultiStepModelRunner
        mindie_config = {
            "backend_type": "atb",
            "model_id": kwargs.get("model_config").model,
            "rank": kwargs.get("rank"),
            "local_rank": kwargs.get("local_rank"),
            "world_size": kwargs.get("parallel_config").world_size,
            "npu_device_id": kwargs.get("local_rank"),
            "inference_mode": 2 if kwargs.get("scheduler_config").chunked_prefill_enabled else 0,
        }
        self.model_runner = MultiStepNPUModelRunner(
            base_model_runner,
            base_model_runner.model_config,
            base_model_runner.parallel_config,
            base_model_runner.scheduler_config,
            base_model_runner.device_config,
            base_model_runner.cache_config,
            load_config=base_model_runner.load_config,
            lora_config=self.lora_config,
            mindie_config=mindie_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=base_model_runner.is_driver_worker,
            prompt_adapter_config=base_model_runner.prompt_adapter_config,
            observability_config=base_model_runner.observability_config,
        )

        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size
        self.multi_step_states: List[Optional[MultiStepState]] = [None] * pipeline_parallel_size
        self.temp_output = None

    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[Tuple[StatefulModelInput, WorkerInput, Dict[str, torch.Tensor]]]:
        """
        Depending on the current state of the request and multi step worker,
        this method may skip the normal _prepare_model_input and
        _prepare_worker_input methods and instead used cached values.
        """
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    broadcast_tensor_dict({}, src=0)
                return None

            virtual_engine = execute_model_req.virtual_engine
            (model_input, worker_input, kwargs) = self._get_driver_input_and_broadcast(execute_model_req)
            assert isinstance(model_input, StatefulModelInput)
            if execute_model_req.is_first_multi_step:
                # cache the worker input and model input for the next steps
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input=worker_input, model_input=model_input
                )
        # if TP workers
        else:
            broadcast_data = self._get_worker_input_from_broadcast()
            # if the driver has sent an empty input, we should stop the worker
            # loop
            if broadcast_data is None:
                return None
            model_input, worker_input, kwargs = broadcast_data
            assert isinstance(model_input, StatefulModelInput)
            virtual_engine = worker_input.virtual_engine
            if model_input.is_first_multi_step:
                pass
                # TODO(will) Can cache the worker input and model input for the
                # next steps. See below for details
            else:
                # TODO(will) possible to also cache and reuse the cached worker
                # input and model input. The idea is essentially the delta
                # optimization for model_inputs. Where the TP workers can cache
                # the model input states and we only broadcast the delta need
                # for the next step (sampled_token_ids from the previous step)

                assert isinstance(model_input, StatefulModelInput)
                # we need to update the last sampled token ids in the model
                # input for the workers so that they can run inplace
                # advance_step
                model_input.add_sampler_output(
                    SamplerOutput(outputs=[], sampled_token_ids=None), model_input.last_sampled_token_ids
                )

        assert model_input is not None
        assert worker_input is not None
        return model_input, worker_input, kwargs

    def _get_worker_input_from_broadcast(
        self
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
            str, torch.Tensor]]]:
        """ Get the worker input from the broadcasted tensor dict. """
        assert self.do_metadata_broadcast
        assert not self.is_driver_worker
        broadcast_data = broadcast_tensor_dict(src=0)
        if not broadcast_data:
            return None

        worker_input = WorkerInput.from_broadcasted_tensor_dict(broadcast_data)
        model_input = (
            self.model_runner.make_model_input_from_broadcasted_tensor_dict(
                broadcast_data))

        kwargs = extract_previous_hidden_states(broadcast_data)

        return model_input, worker_input, kwargs

    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]:
        """
        Get the driver input and broadcast it to other workers.
        """
        assert self.is_driver_worker
        virtual_engine = execute_model_req.virtual_engine
        is_first_multi_step = execute_model_req.is_first_multi_step
        if is_first_multi_step:
            # on first step we prepare the worker input and model input normally
            worker_input: WorkerInput = self.prepare_worker_input(execute_model_req=execute_model_req)
            model_input: StatefulModelInput = self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.virtual_engine,
                execute_model_req.finished_requests_ids,
            )

            if execute_model_req.async_callback:
                model_input.frozen_model_input = dataclasses.replace(  # type: ignore
                    model_input.frozen_model_input, async_callback=execute_model_req.async_callback
                )
        else:
            # on subsequent steps we reuse the worker input and model input
            multi_step_state = self.multi_step_states[virtual_engine]
            worker_input = multi_step_state.worker_input
            model_input = multi_step_state.model_input
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None
            assert frozen_model_input.attn_metadata is not None
            # clear the cached decode metadata so that it can be recomputed on
            # the workers
            frozen_model_input.attn_metadata._cached_decode_metadata = None

        model_input.is_first_multi_step = is_first_multi_step
        model_input.is_last_step = execute_model_req.is_last_step

        if not is_first_multi_step:
            # we broadcast the last sampled token ids to all TP workers so they
            # can update their model input metadata in-place.
            self._prepare_last_sampled_token_ids_for_tp_workers(
                execute_model_req=execute_model_req, model_input=model_input
            )

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_tensor_dict(broadcast_data, src=0)

        # Retuning empty dict here to keep this compatible with
        # `LocalOrDistributedWorkerBase._get_driver_input_and_broadcast`
        return model_input, worker_input, {}

    def _prepare_last_sampled_token_ids_for_tp_workers(
        self,
        execute_model_req: ExecuteModelRequest,
        model_input: StatefulModelInput,
    ) -> None:
        """
        Prepare the last sampled token ids for TP workers. If it's the last
        PP rank, then the last sampled token ids are already in the model_input.
        If it is NOT the last PP rank, then we need to get the last sampled
        token that is cached in the execute_model_req.
        """
        if get_pp_group().is_last_rank:
            assert model_input.cached_outputs[-1].sampler_output.sampled_token_ids is None
            assert model_input.cached_outputs[-1].sampled_token_ids is not None
            model_input.last_sampled_token_ids = model_input.cached_outputs[-1].sampled_token_ids
            # free sampled token ids from the previous step if it has been
            # pythonized. Cannot free the last sampled token ids because
            # we need it for NPU advance_step.
            for output in model_input.cached_outputs[:-1]:
                if output.pythonized:
                    output.sampled_token_ids = None
        else:
            # otherwise we need to get the cached sampled token ids from the
            # execute_model_req
            assert execute_model_req.last_sampled_token_ids is not None
            model_input.last_sampled_token_ids = execute_model_req.last_sampled_token_ids.cuda()
            model_input.add_sampler_output(
                SamplerOutput(outputs=[], sampled_token_ids=None), model_input.last_sampled_token_ids
            )

            # free sampled token ids from the previous step.
            # TODO(will) we could reuse the sampled token ids tensor from
            # the previous step instead.
            for output in model_input.cached_outputs[:-1]:
                output.sampled_token_ids = None
            assert model_input.cached_outputs[-1].sampled_token_ids is not None