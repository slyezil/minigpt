import torch

from src.model import RMSNorm


def test_rmsnorm_preserves_shape():
    norm = RMSNorm(d_model=768)

    x = torch.randn(2, 16, 768)
    output = norm(x)

    assert output.shape == x.shape


def test_rmsnorm_matches_manual_calculation():
    norm = RMSNorm(d_model=4, eps=1e-6)

    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ]
    )

    output = norm(x)

    rms = torch.rsqrt(
        x.pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )

    expected = x * rms

    assert torch.allclose(output, expected, atol=1e-5)