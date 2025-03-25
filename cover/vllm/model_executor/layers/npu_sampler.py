# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# Part of codes in this file was copied from project [vLLM Team][vllm]
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mindie_llm.text_generator.utils.sampling_metadata import (
    SamplingData,
    SamplingParam,
)
from vllm.model_executor.layers.sampler import (
    SamplerOutput,
    get_logprobs,
    _get_sampled_logprob_if_needed,
    _build_sampler_output,
)
from vllm.model_executor.sampling_metadata import (
    SamplingMetadata,
    SequenceGroupToSample,
)
from vllm.sampling_params import SamplingType
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs

# from loguru import logger
_SAMPLING_EPS = 1e-5

SampleResultType = List[Tuple[List[int], List[int]]]


# TODO: Figure out how to remove _get_logprobs
def _to_tensor(data, dtype=None):
    if dtype:
        return torch.tensor(data, dtype=dtype, device=torch.device("npu"))
    else:
        return torch.tensor(data, device=torch.device("npu"))


def _get_ranks(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the ranks of the chosen tokens in a logprob tensor.

    Args:
        x (torch.Tensor): 2D logprob tensor of shape (N, M)
                        where N is the no. of tokens and M is the vocab dim.
        indices (torch.Tensor): List of chosen token indices.

    Returns:
        torch.Tensor: 1D tensor of shape (N,) where N is the no. of tokens.
                    Each element in the returned tensor represents the rank
                    of the chosen token in the input logprob tensor.
    """
    vals = x[
        torch.arange(0, len(x), device=x.device, dtype=indices.dtype), indices
    ]
    return (x > vals[:, None]).long().sum(1).add_(1)


def _get_prompt_logprob_if_needed(
    seq_group: SequenceGroupToSample,
    selected_logprobs: torch.Tensor,
    ranks: torch.Tensor,
    top_token_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    selected_logprobs_idx: int,
    top_logprob_idx: int,
):
    """Compute the prompt logprob from a sequence group if needed."""
    sampling_params = seq_group.sampling_params
    is_prompt = seq_group.is_prompt

    # Find prompt logprobs
    prompt_logprobs: Optional[PromptLogprobs] = None
    if is_prompt and sampling_params.prompt_logprobs is not None:
        prompt_logprobs = []
        num_logprobs = sampling_params.prompt_logprobs
        next_prompt_tokens = _get_next_prompt_tokens(seq_group)
        for token_id in next_prompt_tokens:
            # Calculate the prompt logprob of the real prompt tokens.
            # Use tuple here for performance (to use to_list()).
            # {token_id: (logprob, rank_from_vocab)}
            prompt_logprobs_dict: Dict[int, Tuple[float, int]] = {
                token_id: (
                    selected_logprobs[selected_logprobs_idx].item(),
                    ranks[selected_logprobs_idx].item(),
                )
            }

            # Add top K prompt logprobs along with its rank.
            if num_logprobs > 0:
                prompt_logprobs_dict.update(
                    zip(
                        top_token_ids[top_logprob_idx, :num_logprobs].tolist(),
                        zip(
                            top_logprobs[
                                top_logprob_idx, :num_logprobs
                            ].tolist(),
                            # This is ranks. Since top_logprob is sorted,
                            # we can just use a range here.
                            range(1, num_logprobs + 1),
                        ),
                    )
                )
            prompt_logprobs.append(
                {
                    token_id: Logprob(*logprob_and_rank)
                    for token_id, logprob_and_rank in prompt_logprobs_dict.items()
                }
            )
            # + 1 to go to the next prompt token.
            top_logprob_idx += 1
            selected_logprobs_idx += 1
    return prompt_logprobs, top_logprob_idx, selected_logprobs_idx


def _get_next_prompt_tokens(seq_group: SequenceGroupToSample) -> List[int]:
    """Get a list of next prompt tokens to compute logprob from a
        given sequence group.

    It is used to compute prompt logprob. Imagine you have logprob for each
    query token. Query token needs to know the next prompt token id to compute
    prompt logprob. This is a helper to obtain next prompt token ids.

    This API has to be used only when the caller knows seq_group is in prefill
    stage.

    Returns:
        A list of next prompt tokens to compute logprob.
    """
    assert (
        seq_group.is_prompt
    ), "Caller should ensure the sequence group is in a prefill stage."
    seq_ids = seq_group.seq_ids
    query_len = seq_group.query_len
    assert query_len is not None
    # prompt has only 1 seq id.
    assert len(seq_ids) == 1
    seq_data = seq_group.seq_data[seq_ids[0]]
    computed_len = seq_data.get_num_computed_tokens()
    prompt_tokens = seq_data.prompt_token_ids
    # +1 because we are looking for a next prompt token.
    next_token_index_start = computed_len + 1
    next_token_index_end = min(computed_len + query_len + 1, len(prompt_tokens))
    next_prompt_tokens = prompt_tokens[
        next_token_index_start:next_token_index_end
    ]
    return next_prompt_tokens


class MindIESampler(nn.Module):
    """
    A sampler class for generating tokens using the MindIE Sampler.

    This class performs sampling over token logits, generating tokens based on the
    sampling configurations defined in `sampling_metadata`. It integrates with the
    `mindie_model`, a token generation model, to handle different sampling strategies
    such as greedy, random, and beam search (although beam search is not yet implemented).

    Attributes:
        mindie_model (GeneratorTorch): Integrate MindIE model initialized with the
          configuration `mindie_config` and call the model's `sample` method to generate tokens, it handles the core sampling logic for generating the next token in the sequence.

        include_gpu_probs_tensor (bool): Flag indicating whether to include GPU-based
            probabilities in the returned output tensor.

    Methods:
        forward:
            Performs token sampling and return the results including log probabilities and
              sampled tokens based on the provided logits and sampling metadata.

        construct_data:
            Constructs the necessary data and parameters for sampling based on the provided
              metadata, including configuration for temperature, penalties, and sampling type.

        recover_data:
            Post-processes the sampled tokens and log probabilities, categorizing the results
              according to the sampling types (Greedy, Random). It also constructs the final
              sample results and optionally includes GPU-based probabilities if requested.
    """

    def __init__(self, mindie_model):
        """
        Initializes the MindIESampler with the given configuration and optional GPU probability flag.

        Args:
            mindie_config (MindIESamplerConfig): Configuration object containing the parameters
              for the MindIE model.

            include_gpu_probs_tensor (bool, optional): If set to True, the method will include
              GPU-based probabilities in the returned output tensor. Default is False.

        """
        super().__init__()
        self.mindie_model = mindie_model
        self.include_gpu_probs_tensor = False

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """
        Performs token sampling based on the provided logits and sampling metadata.

        This method uses the `mindie_model` to generate token samples from the logits and metadata.
        The generated tokens and their associated log probabilities are returned as a result.

        Args:
            logits (torch.Tensor): A tensor containing the logits for token generation.
                This tensor should be of shape `(seq_length, vocab_size)`.

            sampling_metadata (SamplingMetadata): Metadata containing information about 
                the sampling configuration, including sampling types, sequence groups, 
                and other sampling parameters.

        Returns:
            Optional[SamplerOutput]: The output of the token sampling process, which contains
            the sampled tokens and their associated log probabilities. 
        """
        _, vocab_size = logits.shape
        expanded_logits_lst = []
        idx = 0
        for seq_group in sampling_metadata.seq_groups:
            best_of = seq_group.sampling_params.best_of
            num_seqs = len(seq_group.seq_ids)
            seq_group_logits = logits[idx:idx + num_seqs]
            if seq_group.is_prompt:
                if seq_group_logits.dim() == 1:
                    seq_group_logits = seq_group_logits.unsqueeze(0)
                expanded_logits = seq_group_logits.repeat_interleave(
                    best_of, dim=0
                )
            else:
                expanded_logits = seq_group_logits
            expanded_logits_lst.append(expanded_logits)
            idx += num_seqs
        expanded_logits = torch.cat(expanded_logits_lst, dim=0)

        mindie_sampling_data, mindie_sampling_param = self.construct_data(
            sampling_metadata, vocab_size
        )
        probs = torch.softmax(expanded_logits, dim=-1, dtype=torch.float)
        logprobs = torch.log_softmax(expanded_logits, dim=-1, dtype=torch.float)
        if mindie_sampling_param:
            sampling_mask = (
                mindie_sampling_param.do_sample_meta.do_sample_tensor.tolist()
            )
        else:
            sampling_mask = [
                seq_group.do_sample
                for seq_group in sampling_metadata.seq_groups
            ]

        filtered_logits = expanded_logits[sampling_mask]

        if filtered_logits.size(0) > 0:
            next_tokens, _ = self.mindie_model.sample(
                filtered_logits,
                sampling_data=mindie_sampling_data,
                sampling_param=mindie_sampling_param,
            )
        else:
            next_tokens = None

        sample_results, maybe_sampled_tokens_tensor = recover_data(
            sampling_metadata=sampling_metadata,
            sampled_tokens=next_tokens,
            logprobs=logprobs,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
        )
        if self.include_gpu_probs_tensor:
            if maybe_sampled_tokens_tensor is None:
                raise RuntimeError("maybe_sampled_tokens_tensor is None")
            on_device_tensors = (probs, logprobs, maybe_sampled_tokens_tensor)
        else:
            on_device_tensors = None

        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = get_logprobs(
            logprobs, sampling_metadata, sample_results
        )
        return _build_sampler_output(
            sample_results,
            sampling_metadata,
            prompt_logprobs,
            sample_logprobs,
            on_device_tensors=on_device_tensors,
        )

    def construct_data(
        self,
        sampling_metadata: SamplingMetadata,
        vocab_size: int,
    ) -> Tuple[SamplingData, SamplingParam]:
        all_input_tokens: List[List[int]] = []
        prompt_tokens: List[List[int]] = []
        output_tokens: List[List[int]] = []
        top_ks: List[int] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        min_ps: List[float] = []
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        repetition_penalties: List[float] = []
        sampling_seeds: List[int] = []
        sample_indices: List[int] = []
        do_samples: List[bool] = []  # To Do
        do_penalties = False
        do_top_p_top_k = False
        do_min_p = False
        greedy_flag = False
        non_greedy_flag = False

        if sampling_metadata.seq_groups is None:
            raise RuntimeError(
                "sampling_metadata.seq_group is None, no data received."
            )
        for seq_group in sampling_metadata.seq_groups:
            seq_ids = seq_group.seq_ids
            sampling_params = seq_group.sampling_params
            temperature = sampling_params.temperature
            p = sampling_params.presence_penalty
            f = sampling_params.frequency_penalty
            r = sampling_params.repetition_penalty
            top_p = sampling_params.top_p
            min_p = sampling_params.min_p
            is_greedy = sampling_params.sampling_type == SamplingType.GREEDY
            best_of = sampling_params.best_of
            if not seq_group.is_prompt:
                do_samples.extend([seq_group.do_sample] * len(seq_ids))  # TODO
            else:
                do_samples.extend([seq_group.do_sample] * best_of)
            # seed = sampling_params.seed
            if not is_greedy:
                non_greedy_flag = True
            if is_greedy:
                seed = 0
                greedy_flag = True
            else:
                seed = sampling_params.seed

            # k should not be greater than the vocab size.
            top_k = min(sampling_params.top_k, vocab_size)
            top_k = vocab_size if top_k == -1 else top_k
            if temperature < _SAMPLING_EPS:
                temperature = 1.0
            if not do_top_p_top_k and (
                top_p < 1.0 - _SAMPLING_EPS or top_k != vocab_size
            ):
                do_top_p_top_k = True
            if not do_min_p and min_p > _SAMPLING_EPS:
                do_min_p = True
            if not do_penalties:
                if abs(p) >= _SAMPLING_EPS:
                    do_penalties = True
                elif abs(f) >= _SAMPLING_EPS:
                    do_penalties = True
                elif abs(r - 1.0) >= _SAMPLING_EPS:
                    do_penalties = True

            if (
                seq_group.is_prompt
                and sampling_params.prompt_logprobs is not None
            ):
                # For tokens in the prompt that we only need to get
                # their logprobs
                query_len = seq_group.query_len
                if query_len is None:
                    raise RuntimeError("query_len is None")
                prefill_len = len(seq_group.prompt_logprob_indices)
                temperatures += [temperature] * prefill_len
                sampling_seeds += [seed] * prefill_len
                top_ps += [top_p] * prefill_len
                top_ks += [top_k] * prefill_len
                min_ps += [min_p] * prefill_len
                presence_penalties += [0] * prefill_len
                frequency_penalties += [0] * prefill_len
                repetition_penalties += [1] * prefill_len
                prompt_tokens.extend([] for _ in range(prefill_len))
                output_tokens.extend([] for _ in range(prefill_len))
                all_input_tokens.extend([] for _ in range(prefill_len))

            if seq_group.do_sample:
                sample_lens = len(seq_group.sample_indices)
                if sample_lens != len(seq_ids):
                    raise ValueError("sample_lens != len(seq_ids)")
                for seq_id in seq_ids:
                    if seq_group.is_prompt:
                        seq_data = seq_group.seq_data[seq_id]
                        prompt_tokens.extend(
                            [seq_data.prompt_token_ids] * best_of
                        )
                        output_tokens.extend(
                            [seq_data.output_token_ids] * best_of
                        )
                        all_input_tokens.extend(
                            [
                                seq_data.prompt_token_ids
                                + seq_data.output_token_ids
                            ]
                            * best_of
                        )
                        if seed is None:
                            lo, hi = (
                                torch.iinfo(torch.long).min,
                                torch.iinfo(torch.long).max,
                            )
                            seeds = [
                                random.randint(lo, hi) for _ in range(best_of)
                            ]
                        else:
                            seeds = [seed] * best_of
                        temperatures += [temperature] * best_of
                        sampling_seeds += seeds
                        top_ps += [top_p] * best_of
                        top_ks += [top_k] * best_of
                        min_ps += [min_p] * best_of
                        presence_penalties += [p] * best_of
                        frequency_penalties += [f] * best_of
                        repetition_penalties += [r] * best_of

                    else:
                        seq_data = seq_group.seq_data[seq_id]
                        prompt_tokens.append(seq_data.prompt_token_ids)
                        output_tokens.append(seq_data.output_token_ids)
                        all_input_tokens.append(
                            seq_data.prompt_token_ids
                            + seq_data.output_token_ids
                        )
                        if seed is None:
                            lo, hi = (
                                torch.iinfo(torch.long).min,
                                torch.iinfo(torch.long).max,
                            )
                            seeds = [random.randint(lo, hi)]
                        else:
                            seeds = [seed]
                        temperatures += [temperature]
                        sampling_seeds += seeds
                        top_ps += [top_p]
                        top_ks += [top_k]
                        min_ps += [min_p]
                        presence_penalties += [p]
                        frequency_penalties += [f]
                        repetition_penalties += [r]

        repetition_penalties = np.array(repetition_penalties, dtype=np.float32)
        frequency_penalties = np.array(frequency_penalties, dtype=np.float32)
        presence_penalties = np.array(presence_penalties, dtype=np.float32)
        temperatures = np.array(temperatures, dtype=np.float32)
        top_ks = np.array(top_ks, dtype=np.int32)
        top_ps = np.array(top_ps, dtype=np.float32)
        sampling_seeds = np.array(sampling_seeds)
        do_samples = np.array(do_samples)

        max_tokens_len = max(
            [len(tokens) for tokens in all_input_tokens], default=0
        )
        # TODO: tokens are tuple now
        padded_all_input_tokens = [
            list(tokens) + [vocab_size] * (max_tokens_len - len(tokens))
            for tokens in all_input_tokens
        ]
        padded_all_input_tokens = np.array(
            padded_all_input_tokens, dtype=np.int32
        )
        output_max_len = max(
            [len(tokens) for tokens in output_tokens], default=0
        )
        # TODO: tokens are tuple now
        padded_output_tokens = [
            list(tokens) + [vocab_size] * (output_max_len - len(tokens))
            for tokens in output_tokens
        ]
        padded_output_tokens = np.array(padded_output_tokens, dtype=np.int32)

        all_input_ids_tensor = (
            _to_tensor(padded_all_input_tokens, torch.int32)
            if padded_all_input_tokens is not None
            else None
        )
        output_ids_tensor = (
            _to_tensor(padded_output_tokens, torch.int32)
            if padded_output_tokens is not None
            else None
        )
        mindie_sampling_data = SamplingData(
            all_input_ids=all_input_ids_tensor, output_ids=output_ids_tensor
        )

        if not non_greedy_flag:
            mindie_sampling_param = None
        else:
            mindie_sampling_param = SamplingParam.from_numpy(
                repetition_penalty=repetition_penalties,
                frequency_penalty=frequency_penalties,
                presence_penalty=presence_penalties,
                temperature=temperatures,
                top_k=top_ks,
                top_p=top_ps,
                seed=sampling_seeds,
                do_sample=do_samples,
                to_tensor=_to_tensor,
            )
        return (mindie_sampling_data, mindie_sampling_param)


def recover_data(
    sampling_metadata: SamplingMetadata,
    sampled_tokens: np.ndarray,
    logprobs: torch.Tensor,
    include_gpu_probs_tensor: bool,
) -> Tuple[SampleResultType, Optional[torch.Tensor]]:
    categorized_seq_group_ids: Dict[SamplingType, List[int]] = {
        t: [] for t in SamplingType
    }
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        sampling_params = seq_group.sampling_params
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    sample_metadata = {}

    # Create output tensor for sampled token ids.
    sampled_tokens = sampled_tokens.tolist()
    if include_gpu_probs_tensor:
        sampled_token_ids_tensor = torch.empty(
            logprobs.shape[0], 1, dtype=torch.long, device=logprobs.device
        )
    else:
        sampled_token_ids_tensor = None

    for sampling_type in SamplingType:
        sample_indices = categorized_sample_indices[sampling_type]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue

        seq_group_id = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_id]
        sample_metadata[sampling_type] = (seq_group_id, seq_groups)

    greedy_samples = []
    random_samples = []
    beam_samples = []
    idx = 0
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        if sampling_params.sampling_type == SamplingType.GREEDY:
            for seq_id in seq_ids:
                greedy_samples.extend([sampled_tokens[idx] for i in seq_ids])
                idx += 1
        elif sampling_params.sampling_type in (
            SamplingType.RANDOM,
            SamplingType.RANDOM_SEED,
        ):
            if seq_group.is_prompt:
                for seq_id in seq_ids:
                    random_samples.extend(
                        [
                            sampled_tokens[idx + i]
                            for i in range(sampling_params.best_of)
                        ]
                    )
                    idx += sampling_params.best_of
            else:
                for seq_id in seq_ids:
                    random_samples.append(sampled_tokens[idx])
                    idx += 1
        elif sampling_params.sampling_type == SamplingType.BEAM:
            if seq_group.is_prompt:
                for seq_id in seq_ids:
                    beam_samples.extend(
                        [
                            sampled_tokens[idx + i]
                            for i in range(sampling_params.best_of)
                        ]
                    )
                    idx += sampling_params.best_of
            else:
                for seq_id in seq_ids:
                    beam_samples.append(sampled_tokens[idx])
                    idx += 1

    for sampling_type in SamplingType:
        if sampling_type not in sample_metadata:
            continue
        (seq_group_id, seq_groups) = sample_metadata[sampling_type]
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(seq_groups, greedy_samples)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            sample_results = _random_sample(seq_groups, random_samples)
        elif sampling_type == SamplingType.BEAM:
            sample_results = beam_wrap(seq_groups, beam_samples)
        sample_results_dict.update(zip(seq_group_id, sample_results))

    sample_results = [
        sample_results_dict.get(i, ([], []))
        for i in range(len(sampling_metadata.seq_groups))
    ]
    return sample_results, sampled_token_ids_tensor


def _greedy_sample(
    selected_seq_groups: List[SequenceGroupToSample],
    samples: np.ndarray,
):
    samples_lst = samples
    sample_idx = 0
    results: SampleResultType = []
    for seq_group in selected_seq_groups:
        if not seq_group.do_sample:
            results.append(([], []))
            continue
        seq_ids = seq_group.seq_ids
        num_parent_seqs = len(seq_ids)
        assert num_parent_seqs == 1, "Greedy sampling should have only one seq."
        parent_ids = list(range(num_parent_seqs))
        next_token_ids = [samples_lst[sample_idx]]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _random_sample(
    selected_seq_groups: List[SequenceGroupToSample],
    samples: np.ndarray,
):
    sample_idx = 0
    results: SampleResultType = []
    for seq_group in selected_seq_groups:
        if not seq_group.do_sample:
            results.append(([], []))
            continue

        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        is_prompt = seq_group.is_prompt
        num_parent_seqs = len(seq_ids)
        if is_prompt:
            parent_ids = [0] * sampling_params.best_of
            next_token_ids = samples[
                sample_idx:sample_idx + sampling_params.best_of
            ]
            sample_idx += sampling_params.best_of
        else:
            parent_ids = list(range(num_parent_seqs))
            next_token_ids = samples[sample_idx:sample_idx + num_parent_seqs]
            sample_idx += num_parent_seqs
        results.append((next_token_ids, parent_ids))
    return results


def beam_wrap(
    selected_seq_groups: List[SequenceGroupToSample],
    samples: np.ndarray,
):
    raise ValueError(f"Unsupported sampling type: beam search")