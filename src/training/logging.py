from transformers import TrainerCallback
import trackio as wandb  # type: ignore


def init_wandb(cfg):
    cfg = cfg or {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return None
    return wandb.init(**cfg.get("init", {}))


class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            wandb.log(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        wandb.finish()
