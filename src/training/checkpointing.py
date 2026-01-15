import os
import torch
import yaml

from training.common import ensure_dir


def save_tuned_model(model, output_dir, tuning_cfg):
    ensure_dir(output_dir)
    if tuning_cfg is not None:
        with open(os.path.join(output_dir, "tuning.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(tuning_cfg, f, sort_keys=False)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
        return

    method = "model"
    if isinstance(tuning_cfg, dict):
        method = tuning_cfg.get("method", method)
    filename = f"{method}.pt"
    torch.save(model.state_dict(), os.path.join(output_dir, filename))
