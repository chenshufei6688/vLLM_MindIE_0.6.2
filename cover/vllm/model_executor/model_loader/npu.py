# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# Part of code in this file was copied from project [vLLM Team][vllm] for adapting usage

import contextlib
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch
from atb_llm.utils.initial import NPUSocInfo
from torch import nn
from vllm.attention import AttentionMetadata
from vllm.config import DeviceConfig, LoadConfig, LoadFormat, ModelConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.npu_sampler import MindIESampler
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]


# TODO: Refactor this to other file
class MindIELlmWrapper(nn.Module):
    """
    A wrapper class for the MindIE model. It provides functionality for forward pass, sampling, 
    and model weight loading.

    Attributes:
        mindie_config : Configuration dictionary containing model parameters, 
        rank: Rank of the current device in the distributed setup.
        local_rank : Local rank of the device.
        npu_id: NPU device ID.
        world_size: Total number of devices in the world size.
        mindie_model: Instance of the generator model, initialized with the provided configuration.
        sampler: Sampler instance for token generation.
        dummy_block_num: Number of dummy blocks for cache creation.
    """
    def __init__(self, mindie_config, linear_method=None, lora_config=None):
        """
        Initializes the MindIELlmWrapper with the provided configuration and optional LoRA setup.

        Args:
            mindie_config: Configuration dictionary for the model, including rank, local_rank, world_size, etc.
            linear_method (optional): Method to apply linear transformations, default is None.
            lora_config (optional): Configuration for LoRA adapters, default is None.
        """

        super(MindIELlmWrapper, self).__init__()

        self.mindie_config = mindie_config
        self.rank = mindie_config["rank"]
        self.local_rank = mindie_config["local_rank"]
        self.npu_id = self.local_rank
        self.world_size = mindie_config["world_size"]
        self.mindie_model = None
        self.sampler = None
        self.need_nz = NPUSocInfo().need_nz

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        lora_requests: List[LoRARequest],
    ) -> torch.Tensor:
        """
        Performs the forward pass through the model, applying attention and token generation.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            positions (torch.Tensor): Indicate the position of each token in the input sequence.
            kv_caches (List[KVCache]): List of key-value caches for attention layers.
            attn_metadata (AttentionMetadata): Metadata related to attention mechanisms,including information 
              relevant to the prefill and decode phases.
            intermediate_tensors (optional): Store intermediate states such as hidden states and residuals 
              during model execution, facilitating operations like gradient checkpointing and model 
              parallelism, default is None.
            lora_requests (List[LoRARequest]): List of LoRA requests to apply during forward pass.

        Returns:
            torch.Tensor: Logits or generated token predictions from the model.
        """
        is_prompt = attn_metadata.prefill_metadata is not None

        if kv_caches[0] is None:
            kv_caches, block_tables, slots = self.create_dummy_kv_cache(attn_metadata, input_ids)
        else:
            block_tables = self.create_block_tables(attn_metadata)
            slots = attn_metadata.slot_mapping

        if attn_metadata.prefill_metadata is None:
            input_lengths = attn_metadata.decode_metadata.seq_lens_tensor
            max_seq_len = attn_metadata.decode_metadata.max_seq_len
            query_lens = []
            lm_head_indices = None
        else:
            input_lengths = attn_metadata.prefill_metadata.seq_lens_tensor
            max_seq_len = attn_metadata.prefill_metadata.max_seq_len
            query_start_loc = attn_metadata.prefill_metadata.query_start_loc
            query_lens_tensor = query_start_loc[1:] - query_start_loc[:-1]
            if attn_metadata.decode_metadata is not None:
                input_lengths = torch.cat((input_lengths, attn_metadata.decode_metadata.seq_lens_tensor), dim=0)
                max_seq_len = max(max_seq_len, attn_metadata.decode_metadata.max_seq_len)
                query_lens_tensor = F.pad(query_lens_tensor, (0, attn_metadata.num_decode_tokens), "constant", 1)
            query_lens = query_lens_tensor.tolist()
            lm_head_indices = query_lens_tensor.cumsum(dim=-1) - 1

        if not lora_requests:
            adapter_ids = ["base"] * len(input_lengths)
        else:
            adapter_ids = [lora_request.lora_name if lora_request else "base" for lora_request in lora_requests]

        # TODO: Can MindIE take advantage of intermediate_tensors?
        logits = self.mindie_model.forward_tensor(
            input_ids,
            positions,
            is_prompt,
            kv_caches,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices,
            adapter_ids=adapter_ids,
            q_lens=query_lens,
        )

        return logits

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return hidden_states

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        ...

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """
        Samples tokens from the logits based on the provided sampling metadata.

        Args:
            logits (torch.Tensor): Logits tensor from which tokens will be sampled.
            sampling_metadata (SamplingMetadata): Metadata defining how sampling should be performed.

        Returns:
            Optional[SamplerOutput]: The sampler output from sampling.
        """
        next_tokens = self.sampler(logits, sampling_metadata) # hidden_states is logits
        return next_tokens

    def load_weights(self):
        """
        Loads the weights into the model, initializing the MindIE model and MindIE sampler.
        """
        self.weight_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        self.mindie_model = GeneratorTorch(self.mindie_config)
        self.sampler = MindIESampler(self.mindie_model)

        torch.set_default_dtype(self.weight_dtype)

    # when warmup, create dummy kvcache, block_tables, slot_mapping
    def create_dummy_kv_cache(self, attn_metadata, input_ids):
        """
        Creates a dummy key-value cache for attention during warmup phase.

        Args:
            attn_metadata (AttentionMetadata): Metadata related to attention for the current batch.
            input_ids (torch.Tensor): Input token IDs for the current batch.

        Returns:
            Tuple: A tuple containing the key-value cache, block tables, and slot mappings.
        """        
        dummy_block_size = 128
        max_s = max(attn_metadata.prefill_metadata.seq_lens_tensor)
        max_need_block = math.ceil(max_s / dummy_block_size)
        batch_size = len(attn_metadata.prefill_metadata.seq_lens_tensor)
        self.dummy_block_num = max_need_block * batch_size

        model_runner = self.mindie_model.model_wrapper.model_runner
        if not self.need_nz:
            dummy_kv_cache_shape = (
                self.dummy_block_num,
                dummy_block_size,
                model_runner.num_kv_heads,
                model_runner.head_size
            )
        else:
            dummy_kv_cache_shape = (
                self.dummy_block_num,
                model_runner.num_kv_heads * model_runner.head_size // 16,
                dummy_block_size,
                16
            )
        kv_cache = [
            (
                torch.empty(
                    size=dummy_kv_cache_shape,
                    dtype=self.weight_dtype,
                    device="npu",
                ),
                torch.empty(
                    size=dummy_kv_cache_shape,
                    dtype=self.weight_dtype,
                    device="npu",
                ),
            )
            for _ in range(model_runner.num_layers)
        ]

        block_tables = torch.zeros(batch_size, max_need_block, dtype=int, device="npu")
        slot = [i for i in range(dummy_block_size)]
        slots = []
        warm_up_len = len(input_ids)
        while warm_up_len > 0:
            if warm_up_len > dummy_block_size:
                slots.extend(slot)
                warm_up_len -= dummy_block_size
            else:
                slots.extend(slot[:warm_up_len])
                warm_up_len = 0
        slots = torch.tensor(slots, dtype=torch.long, device="npu")
        return kv_cache, block_tables, slots

    def create_block_tables(self, attn_metadata):
        """
        Creates block tables for attention, based on prefill and decode metadata.
        """
        if attn_metadata.prefill_metadata is None:
            return attn_metadata.decode_metadata.block_tables
        prefill_block_tables = attn_metadata.prefill_metadata.block_tables
        if prefill_block_tables.numel() == 0:
            return torch.tensor([0], dtype=torch.int32, device="npu")
        if attn_metadata.decode_metadata is None:
            return prefill_block_tables

        decode_block_tables = attn_metadata.decode_metadata.block_tables
        pad_size = prefill_block_tables.size(1) - decode_block_tables.size(1)
        if pad_size > 0:
            decode_block_tables = F.pad(decode_block_tables, (0, pad_size), "constant", 0)
        elif pad_size < 0:
            prefill_block_tables = F.pad(prefill_block_tables, (0, -pad_size), "constant", 0)
        return torch.cat((prefill_block_tables, decode_block_tables), dim=0)


def get_architecture_class_name(model_config: ModelConfig) -> str:
    """
    Determines and returns the architecture class name based on the provided model configuration.

    This function checks the architecture type in the model's configuration and adjusts
    the architecture name in case quantization is enabled and the model is of type "MixtralForCausalLM".
    If quantization is enabled and not set to "fp8", the architecture name is updated to "QuantMixtralForCausalLM".

    Args:
        model_config (ModelConfig): The configuration object containing model-specific settings.

    Returns:
        str: The name of the model architecture class.
    """    
    architectures = getattr(model_config.hf_config, "architectures", [])
    if (
        model_config.quantization is not None
        and model_config.quantization != "fp8"
        and "MixtralForCausalLM" in architectures
    ):
        architectures = ["QuantMixtralForCausalLM"]
    return architectures[0]


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def get_model(
    model_config: ModelConfig, device_config: DeviceConfig, load_config: LoadConfig, mindie_config, **kwargs
) -> nn.Module:
    """
    Loads and initializes a model based on the given configuration and prepares it for inference.

    This function instantiates the `MindIELlmWrapper` model with the provided `mindie_config`, 
    and loads the model weights based on the specified `load_config`. It also supports loading 
    LoRA configurations if provided. The model is moved to the appropriate device (e.g., NPU).

    Args:
        model_config (ModelConfig): The configuration object containing model-specific settings.
        device_config (DeviceConfig): The configuration object that defines the device settings (e.g., NPU).
        load_config (LoadConfig): The configuration object that specifies how to load the model weights.
        mindie_config: The configuration for MindIE specific parameters.

    Returns:
        nn.Module: The initialized Mindie model.

    """
    if kwargs.get("lora_config"):
        logger.info(
            "Using LoRA(s) with MindIE backend:\n"
            "Please make sure your '--lora-modules' matches with your 'lora_adapter.json' in the model directory!\n"
            "Current config for LoRA(s): %s",
            kwargs.get("lora_config"),
        )

    with _set_default_torch_dtype(model_config.dtype):
        with torch.device(device_config.device):
            model = MindIELlmWrapper(mindie_config)
        if load_config.load_format == LoadFormat.DUMMY:
            initialize_dummy_weights(model)
        else:
            model.load_weights()
        model = model.npu()

    return model.eval()