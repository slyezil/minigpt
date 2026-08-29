import torch

from src.device import get_device, get_training_dtype


def test_get_device_returns_xpu():
    device = get_device()

    assert device.type == "xpu"


def test_training_dtype_is_bfloat16():
    dtype = get_training_dtype()

    assert dtype == torch.bfloat16