from typing import Any, Dict, Optional, Union
from torch.distributed import ProcessGroup
import torch


def get_dimension_and_size(x):
    if x is not None:
        return len(x), list(x)
    else:
        return 0, []


def get_true_or_false(x):
    if x:
        return 1, [1]
    else:
        return 1, [0]


def get_dimension_and_size_of_single_value(x):
    if x is not None:
        return 1, [int(x)]
    else:
        return 0, []


def get_size_or_none(x: Optional[torch.Tensor]):
    return x.size() if x is not None else None


bool_keys = ["use_cuda_graph"]
single_value_keys = [
    "num_seq_groups",
    "virtual_engine",
    "num_steps",
    "num_prefill_tokens",
    "num_decode_tokens",
    "num_prefills",
    "max_query_len",
    "max_seq_len",
    "max_prefill_seq_len",
    "max_decode_seq_len",
]
tensor_keys = [
    "input_tokens",
    "input_positions",
    "selected_token_indices",
    "slot_mapping",
    "seq_lens_tensor",
    "query_start_loc",
    "seq_start_loc",
    "context_lens_tensor",
    "block_tables",
    "blocks_to_swap_in",
    "blocks_to_swap_out",
    "blocks_to_copy",
]
other_data_keys = [
    "lora_requests",
    "lora_mapping",
    "multi_modal_kwargs",
    "prompt_adapter_requests",
    "prompt_adapter_mapping",
    "request_ids_to_seq_ids",
    "finished_requests_ids",
    "_cached_prefill_metadata",
    "_cached_decode_metadata",
]
metadata_keys = tensor_keys + bool_keys + single_value_keys + ["seq_lens"]
total_key_num = (
    len(metadata_keys) - 1
)  # seq_lens can be obtain through seq_lens_tensor thus doesn't needed to be broadcast
total_size_data_num = 50


def broadcast(input_: torch.Tensor, src: int = 0, group: Optional[ProcessGroup] = None):
    """Broadcast the input tensor."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_
    # Broadcast.
    torch.distributed.broadcast(input_, src=src, group=group)
    return input_


def prepare_dim_and_size_tensor(
    data_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]]
):
    dim_list = []
    size_list = []
    for key in metadata_keys:
        data = data_dict[key]
        if key in bool_keys:
            dim, size = get_true_or_false(data)
        elif key in single_value_keys:
            dim, size = get_dimension_and_size_of_single_value(data)
        elif key == "seq_lens":
            continue
        else:
            data_size = get_size_or_none(data)
            dim, size = get_dimension_and_size(data_size)
        dim_list.append(dim)
        size_list.extend(size)
    assert len(dim_list) == total_key_num, "the length of dim_list is wrong"
    dim_and_size_list = dim_list + size_list
    if len(dim_and_size_list) < total_size_data_num:
        dim_and_size_list += [-1] * (total_size_data_num - len(dim_and_size_list))
    dim_and_size_tensor = torch.tensor(dim_and_size_list, dtype=torch.int, device="npu")
    return dim_and_size_tensor


def concat_tensor_data(data_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]]):
    concat_data_list = []
    for key in tensor_keys:
        data = data_dict[key]
        if data is not None:
            concat_data_list.extend(data.view(-1).tolist())
    concat_data_tensor = torch.tensor(concat_data_list, dtype=torch.int, device="npu")
    return concat_data_tensor


def get_sizedata_and_singlevalues_from_total(dim_and_size_tensor: torch.tensor):
    dim_list = dim_and_size_tensor[:total_key_num].tolist()
    size_list = dim_and_size_tensor[total_key_num:].tolist()
    dim_idx = 0
    idx = 0
    size_dict = {}
    single_value_dict = {}
    for key in metadata_keys:
        if key in bool_keys:
            bool_data = True if size_list[idx] == 1 else False
            single_value_dict[key] = bool_data
        elif key in single_value_keys:
            single_value_data = size_list[idx] if dim_list[dim_idx] > 0 else None
            single_value_dict[key] = single_value_data
        elif key == "seq_lens":
            continue
        else:
            size_data = (
                torch.Size(size_list[idx : idx + dim_list[dim_idx]])
                if dim_list[dim_idx] > 0
                else None
            )
            size_dict[key] = size_data
        idx += dim_list[dim_idx]
        dim_idx += 1

    return size_dict, single_value_dict


def construct_empty_concat_tensor(size_dict):
    total_element_num = 0
    for key in tensor_keys:
        if not (key in size_dict):
            raise ValueError(f"missing key {key} in reveiced size data")
        if size_dict[key]:
            total_element_num += size_dict[key].numel()
    return torch.empty(total_element_num, dtype=torch.int, device="npu")


def get_tensor_dict_from_concat_tensor(concat_tensor: torch.tensor, size_dict):
    tensor_dict = {}
    idx = 0
    for key in tensor_keys:
        data_size = size_dict[key]
        if data_size is not None:
            tensor_dict[key] = concat_tensor[idx : idx + data_size.numel()].view(
                *data_size
            )
            idx += data_size.numel()
        else:
            tensor_dict[key] = None
    return tensor_dict


def ascend_broadcast_data_dict(
    data_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
):
    group = torch.distributed.group.WORLD
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return data_dict

    rank = torch.distributed.get_rank()

    if rank == src:
        other_data_list = []
        pure_data_dict = {}
        for k, v in data_dict.items():
            if k in other_data_keys:
                other_data_list.append((k, v))
            else:
                pure_data_dict[k] = v
        torch.distributed.broadcast_object_list([other_data_list], src=src)
        dim_and_size_tensor = prepare_dim_and_size_tensor(pure_data_dict)
        handle1 = torch.distributed.broadcast(
            dim_and_size_tensor, src=src, group=group, async_op=True
        )
        concat_tensor = concat_tensor_data(pure_data_dict)
        handle2 = torch.distributed.broadcast(
            concat_tensor, src=src, group=group, async_op=True
        )
        async_handles = [handle1, handle2]
        for async_handle in async_handles:
            async_handle.wait()
    else:
        data_dict = {}
        recv = [None]
        torch.distributed.broadcast_object_list(recv, src=src)
        dim_and_size_tensor = torch.empty(
            total_size_data_num, dtype=torch.int, device="npu"
        )
        other_data_list = recv[0]
        handle1 = torch.distributed.broadcast(
            dim_and_size_tensor, src=src, group=group, async_op=True
        )
        handle1.wait()
        size_dict, single_value_dict = get_sizedata_and_singlevalues_from_total(
            dim_and_size_tensor
        )
        concat_tensor = construct_empty_concat_tensor(size_dict)
        handle2 = torch.distributed.broadcast(
            concat_tensor, src=src, group=group, async_op=True
        )
        data_dict.update(single_value_dict)
        for k, v in other_data_list:
            data_dict[k] = v
        handle2.wait()
        tensor_dict = get_tensor_dict_from_concat_tensor(concat_tensor, size_dict)
        data_dict.update(tensor_dict)
        data_dict["seq_lens"] = data_dict["seq_lens_tensor"].tolist()
    return data_dict