from dataclasses import dataclass

import numpy as np

from data.Needle import NeedleCfg, NeedleDataset, NeedleCausalLMCollator


@dataclass
class NeedleTaskConfig:
    vocab_size: int
    seed: int
    train: dict
    eval: dict
    eval_long: dict | None = None


class NeedleTask:
    def __init__(self, cfg: NeedleTaskConfig):
        self.cfg = cfg

    def vocab_size_with_specials(self):
        return self.cfg.vocab_size + 3

    def _needle_cfg(self, split_cfg):
        return NeedleCfg(
            vocab_size=self.cfg.vocab_size,
            max_seq_len=split_cfg["max_seq_len"],
            vary_length=bool(split_cfg.get("vary_length", False)),
            seed=self.cfg.seed,
            seed_offset=split_cfg.get("seed_offset", 0),
        )

    def build_dataset(self, split_cfg):
        needle_cfg = self._needle_cfg(split_cfg)
        return NeedleDataset(split_cfg["n_examples"], needle_cfg)

    def build_train_eval(self):
        train_ds = self.build_dataset(self.cfg.train)
        eval_ds = self.build_dataset(self.cfg.eval)
        eval_long_ds = None
        if self.cfg.eval_long is not None:
            eval_long_ds = self.build_dataset(self.cfg.eval_long)
        return train_ds, eval_ds, eval_long_ds

    def collator(self):
        cfg = NeedleCfg(vocab_size=self.cfg.vocab_size, max_seq_len=3)
        return NeedleCausalLMCollator(cfg)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        labels = labels.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])

        mask = labels != -100
        if mask.sum() == 0:
            return {"accuracy": 0.0}

        pred_ids = np.argmax(logits[mask], axis=-1)
        acc = (pred_ids == labels[mask]).mean()
        return {"accuracy": float(acc)}


def build_task(cfg):
    name = cfg.get("name", "needle")
    if name != "needle":
        raise ValueError(f"Unsupported task: {name}")

    task_cfg = NeedleTaskConfig(
        vocab_size=cfg["vocab_size"],
        seed=cfg.get("seed", 0),
        train=cfg["train"],
        eval=cfg["eval"],
        eval_long=cfg.get("eval_long"),
    )
    return NeedleTask(task_cfg)
