from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from trl.trainer.dpo_trainer import DPOConfig


@dataclass
class StepwiseDPOConfig(DPOConfig):
    """Configuration class for StepwiseDPOTrainer.
    
    This extends the standard DPOConfig with additional parameters specific to
    step-level optimization.
    """
    
    # Step-level DPO parameters
    step_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter for step-level DPO loss"},
    )
    outcome_beta: float = field(
        default=0.5,
        metadata={"help": "The beta parameter for outcome-level DPO loss"},
    )
    aggregation_strategy: str = field(
        default="weighted_sum",
        metadata={
            "help": "Strategy for aggregating step rewards"
            " (weighted_sum, min_step, harmonic_mean, geometric_mean, custom)"
        },
    )
    step_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Optional weights for each step in weighted_sum strategy"},
    )
    max_steps: int = field(
        default=8,
        metadata={"help": "Maximum number of steps to consider"},
    )
    step_delimiter: str = field(
        default="Step",
        metadata={"help": "String used to identify steps in the model output"},
    )
    
    # Step reward model parameters
    use_external_reward_model: bool = field(
        default=False,
        metadata={"help": "Whether to use an external reward model for step-level rewards"},
    )
    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name or path of the external reward model"},
    )
    
    # Advanced training parameters
    step_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for step-level loss in the total loss"},
    )
    outcome_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for outcome-level loss in the total loss"},
    )
    use_step_wise_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use step-wise prompting during training"},
    )
    step_wise_prompt_template: Optional[str] = field(
        default=None,
        metadata={"help": "Template for step-wise prompting"},
    )
    
    # Evaluation parameters
    evaluate_step_accuracy: bool = field(
        default=True,
        metadata={"help": "Whether to evaluate step-level accuracy during evaluation"},
    )
    step_accuracy_metric: str = field(
        default="exact_match",
        metadata={
            "help": "Metric for evaluating step-level accuracy"
            " (exact_match, f1, semantic_similarity)"
        },
    )