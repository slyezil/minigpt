import torch
import torch.nn.functional as F


def next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    vocab_size = logits.size(-1)

    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )

def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(inputs)

    loss = next_token_loss(
        logits,
        targets,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
    )

    optimizer.step()

    return float(loss.detach().item())