from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class NeedleCfg:
    vocab_size: int      # common tokens: 0..vocab_size-1
    max_seq_len: int     # maximum length of example (including COPY and PASTE), minimum 3
    vary_length: bool = False
    seed: int = 0
    seed_offset: int = 0  # to make two calls to make_dataloader give different datasets

    @property
    def copy_id(self) -> int:
        return self.vocab_size

    @property
    def paste_id(self) -> int:
        return self.vocab_size + 1

    @property
    def pad_id(self) -> int:
        return self.vocab_size + 2

    @property
    def d_vocab(self) -> int:
        return self.vocab_size + 3


class NeedleDataset(Dataset):
    """
    Returns (x, y):
      x: LongTensor [L]  = [random...] COPY needle [random...] PASTE
      y: LongTensor []   = needle
    L:
      - vary_length=False -> always max_seq_len
      - vary_length=True  -> uniformly from 3 to max_seq_len
    Deterministic by idx (convenient to verify).
    """
    def __init__(self, n_examples: int, cfg: NeedleCfg):
        if cfg.max_seq_len < 3:
            raise ValueError("max_seq_len must be >= 3 (COPY, needle, PASTE)")
        self.n = n_examples
        self.cfg = cfg

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        cfg = self.cfg

        g = torch.Generator()
        g.manual_seed(cfg.seed + cfg.seed_offset + idx)

        if cfg.vary_length:
            seq_len = int(torch.randint(3, cfg.max_seq_len + 1, (1,), generator=g).item())
        else:
            seq_len = cfg.max_seq_len

        # common tokens (0..vocab_size-1)
        x = torch.randint(0, cfg.vocab_size, (seq_len,), generator=g, dtype=torch.long)

        # the last real token — PASTE
        x[seq_len - 1] = cfg.paste_id

        # COPY is placed so that after it there is a needle, and PASTE remains at the end
        copy_pos = int(torch.randint(0, seq_len - 2, (1,), generator=g).item())
        needle = int(torch.randint(0, cfg.vocab_size, (1,), generator=g).item())

        x[copy_pos] = cfg.copy_id
        x[copy_pos + 1] = needle

        y = torch.tensor(needle, dtype=torch.long)
        return x, y


class PadCollate:
    """
    Pad right to max_len in batch.
    Returns:
      input_ids:      [B, T]
      labels:         [B]
      attention_mask: [B, T]  (1 on real tokens)
      lengths:        [B]     (real lengths)
    """
    def __init__(self, cfg: NeedleCfg):
        self.cfg = cfg

    def __call__(self, batch):
        xs, ys = zip(*batch)

        lengths = [int(x.numel()) for x in xs]
        T = max(lengths)
        B = len(xs)

        input_ids = torch.full((B, T), self.cfg.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, T), dtype=torch.long)

        for i, x in enumerate(xs):
            L = int(x.numel())
            input_ids[i, :L] = x
            attention_mask[i, :L] = 1

        labels = torch.stack(ys, dim=0)
        lengths = torch.tensor(lengths, dtype=torch.long)
        return input_ids, labels, attention_mask, lengths


class NeedleCausalLMCollator:
    """
    Collate for Needle training.

    Returns:
      input_ids:      [B, T]
      attention_mask: [B, T]
      labels:         [B]
      lengths:        [B]
    """
    def __init__(self, cfg: NeedleCfg):
        self._pad = PadCollate(cfg)

    def __call__(self, batch):
        input_ids, labels, attention_mask, lengths = self._pad(batch)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "lengths": lengths,
        }


def make_dataloader(
    *,
    n_examples: int,
    batch_size: int,
    cfg: NeedleCfg,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = NeedleDataset(n_examples, cfg)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=PadCollate(cfg),
        pin_memory=True,
    )
