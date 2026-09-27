import torch
import torch.nn as nn

from src.benchmark import (
    benchmark_naive_generation,
    print_benchmark,
)


class FakeModel(nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        logits = torch.zeros(
            batch_size,
            seq_len,
            self.vocab_size,
            device=input_ids.device,
        )

        logits[:, :, 7] = 1.0

        return logits


def test_benchmark_naive_generation():
    model = FakeModel()

    prompt_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )

    result = benchmark_naive_generation(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=5,
        warmup_runs=1,
        use_autocast=False,
    )

    print_benchmark(result)

    assert result.prompt_tokens == 4
    assert result.generated_tokens == 5
    assert result.total_tokens == 9
    assert len(result.token_latencies_ms) == 5
    assert result.total_generation_seconds > 0
    assert result.tokens_per_second > 0
    assert result.output_ids.shape == (1, 9)
    assert torch.all(result.output_ids[0, -5:] == 7)