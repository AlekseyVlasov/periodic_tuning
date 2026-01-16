import torch


def get_logits(out):
    if hasattr(out, "logits"):
        return out.logits
    return out


def paste_pos_for_model(model, lengths: torch.Tensor) -> torch.Tensor:
    t = lengths - 1

    if hasattr(model, "period") and hasattr(model, "n_prompt"):
        k = int(model.period)
        p = int(model.n_prompt)
        block = t // k
        offset = t % k
        return block * (k + p) + p + offset

    if hasattr(model, "n_prompt") and hasattr(model, "prompt_embed"):
        return t + int(model.n_prompt)

    return t


def logits_at_paste(logits: torch.Tensor, paste_pos: torch.Tensor) -> torch.Tensor:
    b_idx = torch.arange(logits.size(0), device=logits.device)
    return logits[b_idx, paste_pos, :]
