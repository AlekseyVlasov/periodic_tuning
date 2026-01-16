import trackio as wandb  # type: ignore


def init_wandb(cfg):
    cfg = cfg or {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return None
    return wandb.init(**cfg.get("init", {}))
