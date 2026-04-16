import contextlib
import os

import torch


def _load_torch_npu():
    try:
        import torch_npu  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def device_type():
    if torch.cuda.is_available():
        return "cuda"
    if _load_torch_npu():
        return "npu"
    return "cpu"


def backend():
    current = device_type()
    if current == "cuda":
        return "nccl"
    if current == "npu":
        return "hccl"
    return "gloo"


def default_device(index=None):
    current = device_type()
    if current == "cpu":
        return torch.device("cpu")
    if index is None:
        index = int(os.environ.get("LOCAL_RANK", 0))
    return torch.device(f"{current}:{index}")


def set_device(index):
    current = device_type()
    if current == "cuda":
        torch.cuda.set_device(index)
    elif current == "npu":
        torch.npu.set_device(index)


def synchronize():
    current = device_type()
    if current == "cuda":
        torch.cuda.synchronize()
    elif current == "npu":
        torch.npu.synchronize()


def empty_cache():
    current = device_type()
    if current == "cuda":
        torch.cuda.empty_cache()
    elif current == "npu":
        torch.npu.empty_cache()


def ipc_collect():
    current = device_type()
    if current == "cuda":
        torch.cuda.ipc_collect()
    elif current == "npu" and hasattr(torch.npu, "ipc_collect"):
        torch.npu.ipc_collect()


def current_device():
    current = device_type()
    if current == "cuda":
        return torch.cuda.current_device()
    if current == "npu":
        return torch.npu.current_device()
    return "cpu"


def autocast(dtype):
    current = device_type()
    if current == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    if current == "npu":
        if hasattr(torch.npu, "amp") and hasattr(torch.npu.amp, "autocast"):
            return torch.npu.amp.autocast(dtype=dtype)
        return torch.amp.autocast("npu", dtype=dtype)
    return contextlib.nullcontext()


def patch_cuda_api_for_npu():
    if device_type() != "npu":
        return

    torch.cuda.current_device = torch.npu.current_device
    torch.cuda.set_device = torch.npu.set_device
    torch.cuda.synchronize = torch.npu.synchronize
    torch.cuda.empty_cache = torch.npu.empty_cache
    if hasattr(torch.npu, "ipc_collect"):
        torch.cuda.ipc_collect = torch.npu.ipc_collect
    if hasattr(torch.npu, "amp") and hasattr(torch.npu.amp, "autocast"):
        torch.cuda.amp.autocast = torch.npu.amp.autocast
