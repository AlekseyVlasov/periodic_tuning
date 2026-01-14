import torch
import torch.nn as nn

class PromptTuning(nn.Module):
    """
    Prompt-tuning (soft prompt) as a separate nn.Embedding layer

    Requirements:
      - base_model.backbone.embeddings: nn.Embedding(V, D)
      - base_model can forward(inputs_embeds=..., attention_mask=..., labels=...)
    """

    def __init__(self, base_model: nn.Module, n_prompt: int = 20, freeze_base: bool = True):
        super().__init__()
        self.base_model = base_model
        self.n_prompt = int(n_prompt)

        if hasattr(base_model, 'embeddings'):
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
        if hasattr(self.base_model, 'embeddings'):
            x = self.base_model.embeddings(input_ids)  # [B, T, D]
        else:
            x = self.base_model.backbone.embeddings(input_ids)  # [B, T, D]
        B, T, D = x.shape

        # indices 0..P-1 -> [B, P]
        prompt_ids = torch.arange(self.n_prompt, device=x.device).unsqueeze(0).expand(B, -1)
        p = self.prompt_embed(prompt_ids).to(dtype=x.dtype)  # [B, P, D]

        x = torch.cat([p, x], dim=1)  # [B, P+T, D]

        if attention_mask is not None:
            pm = torch.ones((B, self.n_prompt), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([pm, attention_mask], dim=1)

        if labels is not None:
            pl = torch.full((B, self.n_prompt), -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([pl, labels], dim=1)

        return self.base_model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
