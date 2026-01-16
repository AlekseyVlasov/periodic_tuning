import argparse
import os
import sys

from transformers import TrainingArguments

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from training.checkpointing import save_tuned_model
from training.common import load_config, ensure_dir, save_config
from training.logging import init_wandb, WandbCallback, wandb
from transformers import TrainerCallback
from training.models import apply_tuning, build_base_model, is_periodic_tuning
from training.needle_trainer import NeedleTrainer
from training.tasks import build_task
from utils import fix_seed


def build_training_args(cfg, task_cfg):
    train_cfg = cfg["training"]

    return TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=train_cfg.get("num_train_epochs"),
        learning_rate=train_cfg.get("learning_rate"),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        per_device_train_batch_size=task_cfg["train"]["batch_size"],
        per_device_eval_batch_size=task_cfg["eval"]["batch_size"],
        logging_steps=train_cfg.get("logging_steps", 50),
        eval_strategy=train_cfg.get("eval_strategy", train_cfg.get("evaluation_strategy", "epoch")),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        dataloader_num_workers=task_cfg["train"].get("num_workers", 0),
        remove_unused_columns=False,
        report_to=[],
        seed=cfg["seed"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["output_dir"])
    save_config(cfg, os.path.join(cfg["output_dir"], "config.yaml"))

    fix_seed(cfg["seed"])

    task_cfg = cfg["task"]
    task = build_task(task_cfg)

    model_cfg = cfg["model"]
    model = build_base_model(model_cfg, task.vocab_size_with_specials())
    if hasattr(model, "config"):
        model.config.use_cache = False

    tuning_cfg = cfg["tuning"]
    model = apply_tuning(model, tuning_cfg)

    train_ds, eval_ds, eval_long_ds = task.build_train_eval()
    collator = task.collator()

    training_args = build_training_args(cfg, task_cfg)

    wandb_run = init_wandb(cfg.get("wandb"))
    callbacks = []
    if wandb_run is not None:
        callbacks.append(WandbCallback())

    trainer = NeedleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=task.compute_metrics,
        callbacks=callbacks,
    )

    class TrainEvalCallback(TrainerCallback):
        def __init__(self, trainer_ref, train_dataset):
            self.trainer_ref = trainer_ref
            self.train_dataset = train_dataset

        def on_epoch_end(self, args, state, control, **kwargs):
            self.trainer_ref.evaluate(
                eval_dataset=self.train_dataset,
                metric_key_prefix="train",
            )

    trainer.add_callback(TrainEvalCallback(trainer, train_ds))

    trainer.train()

    final_dir = os.path.join(cfg["output_dir"], "final")
    if is_periodic_tuning(model):
        save_tuned_model(model, final_dir, tuning_cfg)
    else:
        trainer.save_model(final_dir)

    if eval_long_ds is not None:
        eval_long_bs = task_cfg.get("eval_long", {}).get("batch_size")
        if eval_long_bs is not None:
            trainer.args.per_device_eval_batch_size = eval_long_bs
        metrics = trainer.evaluate(eval_dataset=eval_long_ds, metric_key_prefix="eval_long")
        if wandb_run is not None:
            wandb.log(metrics, step=trainer.state.global_step)


if __name__ == "__main__":
    main()
