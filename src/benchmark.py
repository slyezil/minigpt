from dataclasses import dataclass
import statistics
import time

import torch


@dataclass
class BenchmarkResult:
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int

    total_generation_seconds: float

    tokens_per_second: float

    first_token_latency_ms: float
    mean_token_latency_ms: float
    p50_token_latency_ms: float
    p95_token_latency_ms: float

    token_latencies_ms: list[float]

    output_ids: torch.Tensor


def synchronize_device(device: torch.device) -> None:
    """
    Synchronize asynchronous device execution before/after timing.

    CPU operations are synchronous, so nothing is required there.
    """

    if device.type == "xpu":
        torch.xpu.synchronize()


def extract_logits(model_output):
    """
    Support a few common model output formats.

    Our MiniGPT currently returns logits directly, but keeping this helper
    makes the benchmark reusable.
    """

    if isinstance(model_output, torch.Tensor):
        return model_output

    if hasattr(model_output, "logits"):
        return model_output.logits

    if isinstance(model_output, (tuple, list)):
        return model_output[0]

    raise TypeError(
        f"Unsupported model output type: {type(model_output)}"
    )


def percentile(values: list[float], percentile_value: float) -> float:
    """
    Simple percentile implementation without requiring NumPy.
    """

    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = int(
        round(
            (len(sorted_values) - 1)
            * percentile_value
        )
    )

    return sorted_values[index]


def benchmark_naive_generation(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 64,
    warmup_runs: int = 3,
    use_autocast: bool = True,
) -> BenchmarkResult:

    if prompt_ids.ndim != 2:
        raise ValueError(
            "prompt_ids must have shape [batch_size, sequence_length]"
        )

    if prompt_ids.shape[0] != 1:
        raise ValueError(
            "Initial benchmark currently expects batch_size=1"
        )

    device = prompt_ids.device

    model.eval()

    #
    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------
    #
    # The first few GPU/XPU calls can include kernel initialization
    # overhead. We do not want that contaminating the benchmark.
    #

    with torch.inference_mode():

        for _ in range(warmup_runs):

            if device.type == "xpu" and use_autocast:

                with torch.autocast(
                    device_type="xpu",
                    dtype=torch.bfloat16,
                ):
                    _ = model(prompt_ids)

            else:

                _ = model(prompt_ids)

        synchronize_device(device)

    #
    # ---------------------------------------------------------
    # Actual benchmark
    # ---------------------------------------------------------
    #

    tokens = prompt_ids.clone()

    token_latencies_ms = []

    with torch.inference_mode():

        for _ in range(max_new_tokens):

            synchronize_device(device)

            start = time.perf_counter()

            if device.type == "xpu" and use_autocast:

                with torch.autocast(
                    device_type="xpu",
                    dtype=torch.bfloat16,
                ):
                    output = model(tokens)

            else:

                output = model(tokens)

            logits = extract_logits(output)

            #
            # Only use the final position to select the next token.
            #
            # IMPORTANT:
            #
            # This does NOT yet optimize the LM head.
            #
            # The model still computed logits for every position.
            #
            # We are only selecting the final position here.
            #

            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

            synchronize_device(device)

            end = time.perf_counter()

            latency_ms = (
                end - start
            ) * 1000.0

            token_latencies_ms.append(
                latency_ms
            )

    total_generation_seconds = (
        sum(token_latencies_ms) / 1000.0
    )

    tokens_per_second = (
        max_new_tokens
        / total_generation_seconds
    )

    first_token_latency_ms = (
        token_latencies_ms[0]
    )

    mean_token_latency_ms = statistics.mean(
        token_latencies_ms
    )

    p50_token_latency_ms = percentile(
        token_latencies_ms,
        0.50,
    )

    p95_token_latency_ms = percentile(
        token_latencies_ms,
        0.95,
    )

    return BenchmarkResult(
        prompt_tokens=prompt_ids.shape[1],
        generated_tokens=max_new_tokens,
        total_tokens=tokens.shape[1],

        total_generation_seconds=total_generation_seconds,

        tokens_per_second=tokens_per_second,

        first_token_latency_ms=first_token_latency_ms,
        mean_token_latency_ms=mean_token_latency_ms,
        p50_token_latency_ms=p50_token_latency_ms,
        p95_token_latency_ms=p95_token_latency_ms,

        token_latencies_ms=token_latencies_ms,

        output_ids=tokens,
    )
def print_benchmark(result: BenchmarkResult) -> None:
    print()
    print("=" * 60)
    print("MiniGPT Inference Benchmark")
    print("=" * 60)

    print(f"Prompt tokens:              {result.prompt_tokens}")
    print(f"Generated tokens:           {result.generated_tokens}")
    print(f"Total sequence tokens:      {result.total_tokens}")

    print("-" * 60)

    print(
        f"Total generation time:      "
        f"{result.total_generation_seconds:.6f} s"
    )

    print(
        f"Generation throughput:      "
        f"{result.tokens_per_second:.2f} tok/s"
    )

    print("-" * 60)

    print(
        f"First token latency:        "
        f"{result.first_token_latency_ms:.2f} ms"
    )

    print(
        f"Mean token latency:         "
        f"{result.mean_token_latency_ms:.2f} ms"
    )

    print(
        f"P50 token latency:          "
        f"{result.p50_token_latency_ms:.2f} ms"
    )

    print(
        f"P95 token latency:          "
        f"{result.p95_token_latency_ms:.2f} ms"
    )

    print("=" * 60)
    print()