from transformers import AutoModelForCausalLM, MambaConfig, MambaForCausalLM
from peft import LoraConfig, get_peft_model

from models.PromptTuning import PromptTuning
from models.PeriodicTuning import PeriodicTuning


def build_base_model(model_cfg, vocab_size):
    pretrained = model_cfg.get("pretrained_name_or_path")
    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(pretrained)
        if vocab_size is not None and model.config.vocab_size != vocab_size:
            model.resize_token_embeddings(vocab_size)
        return model

    mamba_cfg = model_cfg.get("config", {})
    mamba_cfg = dict(mamba_cfg)
    mamba_cfg["vocab_size"] = vocab_size
    config = MambaConfig(**mamba_cfg)
    return MambaForCausalLM(config)


def apply_tuning(model, tuning_cfg):
    method = tuning_cfg.get("method", "full")
    if method == "full":
        return model

    if method == "prompt":
        prompt_cfg = tuning_cfg.get("prompt", {})
        return PromptTuning(
            model,
            n_prompt=prompt_cfg.get("n_prompt", 20),
            freeze_base=bool(prompt_cfg.get("freeze_base", True)),
        )

    if method == "periodic":
        periodic_cfg = tuning_cfg.get("periodic", {})
        return PeriodicTuning(
            model,
            n_prompt=periodic_cfg.get("n_prompt", 20),
            period=periodic_cfg.get("period", 128),
            freeze_base=bool(periodic_cfg.get("freeze_base", True)),
        )

    if method == "lora":
        lora_cfg = dict(tuning_cfg.get("lora", {}))
        if "task_type" not in lora_cfg:
            lora_cfg["task_type"] = "CAUSAL_LM"
        peft_cfg = LoraConfig(**lora_cfg)
        return get_peft_model(model, peft_cfg)

    raise ValueError(f"Unknown tuning method: {method}")


def is_periodic_tuning(model):
    return hasattr(model, "period") and hasattr(model, "prompt_embed") and hasattr(model, "base_model")
