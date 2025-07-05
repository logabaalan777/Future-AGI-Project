import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field

from transformers import PreTrainedModel
from trl import DPOTrainer
from trl.trainer.dpo_trainer import DPOConfig

from .config import StepwiseDPOConfig
from .utils import get_step_rewards, compute_step_loss, extract_steps


class StepwiseDPOTrainer(DPOTrainer):
    """Trainer for Stepwise Direct Preference Optimization (DPO)."""

    def __init__(
        self,
        model: PreTrainedModel,
        args: StepwiseDPOConfig = None,
        processing_class: Optional[Any] = None,
        step_beta: float = 0.1,
        outcome_beta: float = 0.5,
        aggregation_strategy: str = "weighted_sum",
        step_weights: Optional[List[float]] = None,
        max_steps: int = 8,
        step_delimiter: str = "Step",
        reward_model: Optional[PreTrainedModel] = None,
        step_loss_weight: float = 1.0,
        outcome_loss_weight: float = 1.0,
        **kwargs
    ):
        if args is None:
            args = StepwiseDPOConfig()
        
        super().__init__(model=model, args=args, processing_class=processing_class, **kwargs)
        
        self.processing_class = processing_class
        self.step_beta = step_beta
        self.outcome_beta = outcome_beta
        self.aggregation_strategy = aggregation_strategy
        self.step_weights = step_weights
        self.max_steps = max_steps
        self.step_delimiter = step_delimiter
        self.reward_model = reward_model
        self.step_loss_weight = step_loss_weight
        self.outcome_loss_weight = outcome_loss_weight

        self._register_hooks()

    def _register_hooks(self):
        """Register hooks for capturing intermediate activations."""
        pass  # Extend if needed

    def _get_batch_logps(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if labels is None:
            labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss = loss.view(shift_labels.size())
        log_probs = -loss.sum(dim=-1)

        return log_probs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        chosen_inputs = {k: v for k, v in inputs.items() if k.startswith("chosen")}
        rejected_inputs = {k: v for k, v in inputs.items() if k.startswith("rejected")}

        chosen_logps = self._get_batch_logps(
            model=model,
            input_ids=chosen_inputs["chosen_input_ids"],
            attention_mask=chosen_inputs.get("chosen_attention_mask"),
            labels=chosen_inputs.get("chosen_labels"),
        )

        rejected_logps = self._get_batch_logps(
            model=model,
            input_ids=rejected_inputs["rejected_input_ids"],
            attention_mask=rejected_inputs.get("rejected_attention_mask"),
            labels=rejected_inputs.get("rejected_labels"),
        )

        chosen_steps = extract_steps(
            chosen_inputs["chosen_input_ids"],
            self.processing_class,
            self.step_delimiter,
            self.max_steps,
        )

        rejected_steps = extract_steps(
            rejected_inputs["rejected_input_ids"],
            self.processing_class,
            self.step_delimiter,
            self.max_steps,
        )

        step_rewards_chosen = get_step_rewards(
            model,
            chosen_steps,
            self.processing_class,
            strategy=self.aggregation_strategy,
            weights=self.step_weights,
        )

        step_rewards_rejected = get_step_rewards(
            model,
            rejected_steps,
            self.processing_class,
            strategy=self.aggregation_strategy,
            weights=self.step_weights,
        )

        outcome_loss = self._dpo_loss(
            policy_chosen_logps=chosen_logps,
            policy_rejected_logps=rejected_logps,
            reference_chosen_logps=inputs.get("reference_chosen_logps"),
            reference_rejected_logps=inputs.get("reference_rejected_logps"),
            beta=self.outcome_beta,
        )

        step_loss = compute_step_loss(
            chosen_steps_logps=chosen_logps,
            rejected_steps_logps=rejected_logps,
            step_rewards_chosen=step_rewards_chosen,
            step_rewards_rejected=step_rewards_rejected,
            beta=self.step_beta,
        )

        loss = (self.outcome_loss_weight * outcome_loss) + (self.step_loss_weight * step_loss)

        if return_outputs:
            return loss, {"outcome_loss": outcome_loss, "step_loss": step_loss}
        return loss

    def _dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: Optional[torch.FloatTensor] = None,
        reference_rejected_logps: Optional[torch.FloatTensor] = None,
        beta: float = 0.1,
    ) -> torch.FloatTensor:
        if reference_chosen_logps is None:
            reference_chosen_logps = policy_chosen_logps.detach()
        if reference_rejected_logps is None:
            reference_rejected_logps = policy_rejected_logps.detach()

        chosen_ratio = policy_chosen_logps - reference_chosen_logps
        rejected_ratio = policy_rejected_logps - reference_rejected_logps

        logits = beta * (chosen_ratio - rejected_ratio)
        loss = -F.logsigmoid(logits).mean()

        return loss

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.should_save:
            config_dict = {
                "step_beta": self.step_beta,
                "outcome_beta": self.outcome_beta,
                "aggregation_strategy": self.aggregation_strategy,
                "step_weights": self.step_weights,
                "max_steps": self.max_steps,
                "step_delimiter": self.step_delimiter,
                "step_loss_weight": self.step_loss_weight,
                "outcome_loss_weight": self.outcome_loss_weight,
            }

            torch.save(config_dict, f"{output_dir}/stepwise_dpo_config.pt")

            import json
            with open(os.path.join(output_dir, "stepwise_dpo_config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
