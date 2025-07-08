# Stepwise DPO Architecture and Workflow

This document explains the overall architecture and workflow of the Stepwise DPO trainer, based on the "Let's Verify Step by Step" methodology (arXiv:2408.15240v1).

## Architecture Overview

The Stepwise DPO trainer extends Hugging Face's `DPOTrainer` with additional components for step-level reward modeling and aggregation. The architecture consists of the following main components:

1. **Core Trainer**: `StepwiseDPOTrainer` class that handles the training loop and loss computation
2. **Configuration**: `StepwiseDPOConfig` class that defines training parameters
3. **Reward Models**: Classes for evaluating reasoning steps and outcomes
4. **Utility Functions**: Functions for step extraction, reward aggregation, and loss computation

## Component Relationships

```
┌─────────────────────────────────────┐
│           StepwiseDPOTrainer        │
├─────────────────────────────────────┤
│ - model                             │
│ - ref_model                         │
│ - reward_model                      │
│ - tokenizer                         │
│ - args (StepwiseDPOConfig)          │
├─────────────────────────────────────┤
│ + compute_loss()                    │
│ + _dpo_loss()                       │
│ + save_model()                      │
└───────────────┬─────────────────────┘
                │
                │ uses
                ▼
┌─────────────────────────────────────┐
│           Utility Functions         │
├─────────────────────────────────────┤
│ + extract_steps()                   │
│ + get_step_rewards()                │
│ + aggregate_rewards()               │
│ + compute_step_loss()               │
└───────────────┬─────────────────────┘
                │
                │ may use
                ▼
┌─────────────────────────────────────┐
│            Reward Models            │
├─────────────────────────────────────┤
│ - ProcessRewardModel                │
│ - OutcomeRewardModel                │
│ - CombinedRewardModel               │
└─────────────────────────────────────┘
```

## Training Workflow

The Stepwise DPO training workflow involves the following steps:

1. **Data Preparation**: Prepare training data with clear step delimiters
2. **Model Initialization**: Initialize the model, reference model, and optional reward model
3. **Training Loop**: For each batch:
   - Extract steps from chosen and rejected responses
   - Compute log probabilities for each step
   - Calculate rewards for each step
   - Aggregate step rewards
   - Compute step-level and outcome-level losses
   - Combine losses and update model parameters
4. **Evaluation**: Evaluate the model on validation data
5. **Model Saving**: Save the trained model and configuration

## Detailed Workflow

### 1. Data Preparation

Training data should include:
- Prompts
- Chosen (preferred) responses with clear step delimiters
- Rejected (less preferred) responses with clear step delimiters

Example format:
```json
{
  "prompt": "Solve the equation: 3x + 7 = 22",
  "chosen": "Step 1: I'll start by isolating the variable term...",
  "rejected": "Step 1: I need to solve for x in the equation..."
}
```

### 2. Model Initialization

```python
trainer = StepwiseDPOTrainer(
    model=model,                    # Model to train
    ref_model=ref_model,            # Reference model (can be None)
    args=training_args,             # StepwiseDPOConfig
    processing_class=tokenizer,     # Tokenizer or processor for the model
    train_dataset=train_dataset,    # Training data
    eval_dataset=eval_dataset,      # Evaluation data
    reward_model=reward_model,      # Optional external reward model
)```

### 3. Training Loop

The core of the training loop is implemented in the `compute_loss` method:

```python
def compute_loss(self, model, inputs, return_outputs=False):
    # Extract steps from chosen and rejected responses
    chosen_steps = extract_steps(inputs["chosen"], self.args.step_delimiter, self.args.max_steps)
    rejected_steps = extract_steps(inputs["rejected"], self.args.step_delimiter, self.args.max_steps)
    
    # Compute log probabilities for each step
    chosen_step_logps = compute_logps_for_steps(model, chosen_steps, inputs["prompt"])
    rejected_step_logps = compute_logps_for_steps(model, rejected_steps, inputs["prompt"])
    
    # Calculate rewards for each step
    chosen_step_rewards = get_step_rewards(chosen_steps, self.reward_model)
    rejected_step_rewards = get_step_rewards(rejected_steps, self.reward_model)
    
    # Aggregate step rewards
    chosen_agg_reward = aggregate_rewards(chosen_step_rewards, self.args.aggregation_strategy)
    rejected_agg_reward = aggregate_rewards(rejected_step_rewards, self.args.aggregation_strategy)
    
    # Compute step-level and outcome-level losses
    step_loss = compute_step_loss(chosen_step_logps, rejected_step_logps, 
                                 chosen_step_rewards, rejected_step_rewards, self.args.step_beta)
    outcome_loss = self._dpo_loss(inputs["chosen_logps"], inputs["rejected_logps"], 
                                 chosen_agg_reward, rejected_agg_reward, self.args.outcome_beta)
    
    # Combine losses
    loss = self.args.step_loss_weight * step_loss + self.args.outcome_loss_weight * outcome_loss
    
    return loss
```

### 4. Evaluation

Evaluation can include:
- Standard metrics like loss and accuracy
- Step-level metrics such as step accuracy or step reward
- Outcome-level metrics such as overall response quality

### 5. Model Saving

The trained model and its configuration are saved using the `save_model` method:

```python
def save_model(self, output_dir=None, _internal_call=False):
    # Save the model using the parent class method
    super().save_model(output_dir, _internal_call)
    
    # Save Stepwise DPO specific configuration
    if output_dir is None:
        output_dir = self.args.output_dir
        
    stepwise_config = {
        "step_beta": self.args.step_beta,
        "outcome_beta": self.args.outcome_beta,
        "aggregation_strategy": self.args.aggregation_strategy,
        "step_weights": self.args.step_weights,
        "max_steps": self.args.max_steps,
        "step_delimiter": self.args.step_delimiter,
        "step_loss_weight": self.args.step_loss_weight,
        "outcome_loss_weight": self.args.outcome_loss_weight,
    }
    
    # Save as both .pt and .json for better compatibility and inspection
    torch.save(stepwise_config, os.path.join(output_dir, "stepwise_config.pt"))
    with open(os.path.join(output_dir, "stepwise_config.json"), "w") as f:
        json.dump(stepwise_config, f, indent=2)
```

## Key Configuration Parameters

The `StepwiseDPOConfig` class extends Hugging Face's `DPOConfig` with additional parameters for step-level DPO:

```python
class StepwiseDPOConfig(DPOConfig):
    """Configuration class for Stepwise DPO Trainer."""
    
    def __init__(self, 
                 # Step-level DPO parameters
                 step_beta: float = 0.1,
                 outcome_beta: float = 0.1,
                 aggregation_strategy: str = "weighted_sum",
                 step_weights: Optional[List[float]] = None,
                 max_steps: Optional[int] = None,
                 step_delimiter: str = "\n\nStep",
                 # External reward model parameters
                 use_reward_model: bool = False,
                 reward_model_path: Optional[str] = None,
                 # Advanced training parameters
                 step_loss_weight: float = 0.5,
                 outcome_loss_weight: float = 0.5,
                 use_step_wise_prompt: bool = False,
                 step_wise_prompt_template: Optional[str] = None,
                 # Evaluation parameters
                 evaluate_step_accuracy: bool = False,
                 step_accuracy_metric: Optional[str] = None,
                 **kwargs):
        """Initialize StepwiseDPOConfig."""
        super().__init__(**kwargs)
        # Initialize parameters...
```

## Reward Models

The Stepwise DPO trainer supports three types of reward models:

1. **ProcessRewardModel**: Evaluates individual reasoning steps
2. **OutcomeRewardModel**: Evaluates final outcomes
3. **CombinedRewardModel**: Integrates both process and outcome evaluation

These models can be used to provide more sophisticated reward signals than simple log probability ratios.

## Customization Points

The Stepwise DPO trainer provides several customization points:

1. **Aggregation Strategy**: Choose from predefined strategies or implement a custom one
2. **Step Delimiter**: Define how steps are identified in model outputs
3. **Reward Weighting**: Balance between step-level and outcome-level rewards
4. **External Reward Models**: Integrate specialized models for step evaluation

## Usage Examples

### Basic Usage

```python
from trainer import StepwiseDPOTrainer, StepwiseDPOConfig

# Configure the trainer
training_args = StepwiseDPOConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    step_beta=0.1,
    outcome_beta=0.1,
    aggregation_strategy="weighted_sum",
    step_delimiter="\n\nStep",
)

# Initialize and train
trainer = StepwiseDPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

trainer.train()
```

### With Custom Reward Model

```python
from reward_model import CombinedRewardModel

# Create a custom reward model
reward_model = CombinedRewardModel(
    process_model_name="path/to/process/model",
    outcome_model_name="path/to/outcome/model",
    process_weight=0.7,
    outcome_weight=0.3,
)

# Initialize trainer with reward model
trainer = StepwiseDPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    reward_model=reward_model,
)
```

## References

1. "Let's Verify Step by Step" (arXiv:2408.15240v1)
2. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)
3. Process Supervision: Going Beyond Step-by-Step Instructions (Lightman et al., 2023)