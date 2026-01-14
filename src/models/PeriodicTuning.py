import torch
import torch.nn as nn

class PeriodicTuning(nn.Module):
    """
    Periodic-tuning: inserts n_prompt soft-tokens periodically along the sequence.

    If period = K, then prompt is inserted:
      - before tokens 0..K-1 (at the beginning)
      - then before tokens K..2K-1
      - then before tokens 2K..3K-1
      ... and so on.

    Requirements:
      - base_model.embeddings or base_model.backbone.embeddings: nn.Embedding(V, D)
      - base_model can forward(inputs_embeds=..., attention_mask=..., labels=...)
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_prompt: int = 20,
        period: int = 128,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.n_prompt = int(n_prompt)
        self.period = int(period)
        if self.period <= 0:
            raise ValueError("period must be > 0")

        if hasattr(base_model, "embeddings"):
            D = base_model.embeddings.embedding_dim
        else:
            D = base_model.backbone.embeddings.embedding_dim

        self.prompt_embed = nn.Embedding(self.n_prompt, D)
        nn.init.normal_(self.prompt_embed.weight, mean=0.0, std=0.02)

        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False
            for p in self.prompt_embed.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if hasattr(self.base_model, "embeddings"):
            x = self.base_model.embeddings(input_ids)          # [B, T, D]
        else:
            x = self.base_model.backbone.embeddings(input_ids) # [B, T, D]

        B, T, D = x.shape
        K = self.period
        P = self.n_prompt

        # how many blocks of K tokens (the last one may be incomplete)
        n_blocks = (T + K - 1) // K
        Ptot = n_blocks * P
        new_len = T + Ptot

        t = torch.arange(T, device=x.device)
        block = t // K
        offset = t % K
        token_pos = block * (K + P) + P + offset

        b = torch.arange(n_blocks, device=x.device).unsqueeze(1)
        j = torch.arange(P, device=x.device).unsqueeze(0)
        prompt_pos = (b * (K + P) + j).reshape(-1)

        out = x.new_empty((B, new_len, D))


        out[:, token_pos, :] = x

        w = self.prompt_embed.weight.to(dtype=x.dtype, device=x.device)
        w_blocks = w.unsqueeze(0).expand(n_blocks, -1, -1).reshape(Ptot, D)
        out[:, prompt_pos, :] = w_blocks.unsqueeze(0).expand(B, -1, -1)

        if attention_mask is not None:
            new_mask = attention_mask.new_zeros((B, new_len))
            new_mask[:, prompt_pos] = 1
            new_mask[:, token_pos] = attention_mask
            attention_mask = new_mask

        if labels is not None and torch.is_tensor(labels) and labels.dim() == 2:
            new_labels = labels.new_full((B, new_len), -100)
            new_labels[:, token_pos] = labels
            labels = new_labels

        return self.base_model(
            inputs_embeds=out,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
