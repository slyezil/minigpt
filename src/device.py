import torch


def get_device() -> torch.device:
    if not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU is not available. "
            "Make sure PyTorch XPU and a compatible Intel graphics driver are installed."
        )

    return torch.device("xpu")


def get_training_dtype() -> torch.dtype:
    return torch.bfloat16