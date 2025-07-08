# Stepwise DPO Trainer

A PyTorch implementation of Stepwise Direct Preference Optimization (DPO) for training language models with step-level reward aggregation, based on the "Let's Verify Step by Step" methodology (arXiv:2408.15240v1).

## Overview

Stepwise DPO extends Hugging Face's DPO Trainer to optimize individual reasoning steps rather than just the final answer. This approach leads to more reliable and accurate reasoning in language models, particularly for tasks requiring step-by-step problem solving like mathematics, logic, and complex reasoning.

### Key Features

- **Step-level Reward Modeling**: Evaluate and optimize individual reasoning steps
- **Flexible Reward Aggregation**: Multiple strategies for combining step-level rewards
- **HuggingFace Compatibility**: Built on top of the HuggingFace ecosystem
- **Efficient Training**: Optimized for both process and outcome quality
- **Comprehensive Evaluation**: Tools for analyzing step-level and outcome-level performance

## Installation

### Option 1: Install in Development Mode

To install the package in development mode, run:

```bash
# Install in development mode
pip install -e .
```

This will install the package in editable mode, allowing you to make changes to the code without reinstalling.

### Option 2: Add Project Root to Python Path

Alternatively, you can add the project root to your Python path before running scripts:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This is already included in the example scripts.

## Project Structure

```
stepwise-dpo/
├── trainer/                 # Core Stepwise DPO implementation
│   ├── __init__.py          # Package exports
│   ├── stepwise_dpo.py      # Main trainer implementation
│   ├── config.py            # Configuration classes
│   └── utils.py             # Utility functions
├── reward_model/            # Reward model implementations
│   ├── __init__.py          # Package exports
│   ├── process_reward_model.py    # Step-level reward model
│   ├── outcome_reward_model.py    # Outcome-level reward model
│   └── combined_reward_model.py   # Combined reward model
├── data/                    # Sample datasets
│   ├── sample_math_dpo_data.json     # Math problems in JSON format
│   └── sample_gsm8k_dpo_data.jsonl   # GSM8K problems in JSONL format
├── scripts/                 # Training and evaluation scripts
│   ├── train_example.py             # Basic training example
│   ├── train_with_jsonl.py          # Training with JSONL data
│   ├── compare_aggregation_strategies.py  # Compare different strategies
│   ├── prepare_data.py              # Data preparation utilities
├── docs/                    # Documentation
│   ├── architecture.md             # Overall architecture
│   ├── reward_aggregation.md       # Aggregation strategies
│   ├── step_extraction.md          # Step extraction process
│   └── custom_reward_models.md     # Custom reward models
├── README.md                # Project overview
└── requirements.txt         # Dependencies
```

## Usage

### Basic Usage

```python
from trainer import StepwiseDPOTrainer, StepwiseDPOConfig
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "your/model"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dataset with prompts, chosen, and rejected responses
dataset = Dataset.from_dict({
    "prompt": ["Solve: 3x + 7 = 22", ...],
    "chosen": ["Step 1: Subtract 7 from both sides...", ...],
    "rejected": ["Step 1: I'll solve for x...", ...]
})

# Configure the trainer
training_args = StepwiseDPOConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    # Stepwise DPO specific parameters
    step_beta=0.1,
    outcome_beta=0.1,
    aggregation_strategy="weighted_sum",
    step_delimiter="\n\nStep",
    step_loss_weight=0.7,
    outcome_loss_weight=0.3,
)

# Initialize and train
trainer = StepwiseDPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
)

trainer.train()
```

### Command Line Usage

```bash
# Train a model using the provided script
python scripts/train_example.py \
    --model_name_or_path "your/model" \
    --data_path "data/sample_math_dpo_data.json" \
    --output_dir "./results" \
    --num_train_epochs 3 \
    --step_beta 0.1 \
    --outcome_beta 0.1 \
    --aggregation_strategy "weighted_sum"
```

## Reward Aggregation Strategies

The Stepwise DPO trainer supports several reward aggregation strategies:

- **weighted_sum**: Linear combination of step rewards (with uniform or custom weights)
- **min_step**: Minimum reward across all steps (enforces correctness at every step)
- **harmonic_mean**: Gives more weight to lower values (sensitive to poor-performing steps)
- **geometric_mean**: Balanced approach that dampens outliers
- **custom**: Implement your own aggregation function

For more details, see [docs/reward_aggregation.md](docs/reward_aggregation.md).

## Custom Reward Models

You can use custom reward models to provide more sophisticated step evaluation:

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
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_model=reward_model,
)
```

For more details, see [docs/custom_reward_models.md](docs/custom_reward_models.md).

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Reward Aggregation Strategies](docs/reward_aggregation.md)
- [Step Extraction Process](docs/step_extraction.md)
- [Custom Reward Models](docs/custom_reward_models.md)

## Examples

- [Basic Training Example](scripts/train_example.py)
- [Training with JSONL Data](scripts/train_with_jsonl.py)

## References

1. "Let's Verify Step by Step" (arXiv:2408.15240v1)
2. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)
3. Process Supervision: Going Beyond Step-by-Step Instructions (Lightman et al., 2023)

