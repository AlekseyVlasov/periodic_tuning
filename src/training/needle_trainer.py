import torch
import torch.nn.functional as F
from transformers import Trainer

from training.needle_utils import get_logits, paste_pos_for_model, logits_at_paste


class NeedleTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        lengths = inputs.pop("lengths")

        outputs = model(**inputs)
        logits = get_logits(outputs)
        paste_pos = paste_pos_for_model(model, lengths)
        last_logits = logits_at_paste(logits, paste_pos)
        loss = F.cross_entropy(last_logits, labels)

        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        lengths = inputs.pop("lengths")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = get_logits(outputs)
            paste_pos = paste_pos_for_model(model, lengths)
            last_logits = logits_at_paste(logits, paste_pos)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(last_logits, labels)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, last_logits, labels)
