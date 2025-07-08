#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to compare different reward aggregation strategies
in the Stepwise DPO Trainer.

This script loads a small language model and sample data, then trains multiple models
using different reward aggregation strategies and compares their performance.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy

# Import our custom Stepwise DPO trainer
# Add the project root to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import StepwiseDPOTrainer, StepwiseDPOConfig, extract_steps, aggregate_rewards
from reward_model import CombinedRewardModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load sample data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_math_dpo_data.json")
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Convert to HuggingFace dataset
dataset = Dataset.from_dict({
    "prompt": [item["prompt"] for item in raw_data["data"]],
    "chosen": [item["chosen"] for item in raw_data["data"]],
    "rejected": [item["rejected"] for item in raw_data["data"]]
})

# Split into train and eval
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Eval dataset size: {len(dataset['test'])}")

# Load a small model for demonstration purposes
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for demonstration
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Define the aggregation strategies to compare
strategies = [
    "weighted_sum",
    "min_step",
    "harmonic_mean",
    "geometric_mean"
]

# Define custom weights for weighted_sum strategy
custom_weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Emphasize earlier steps

# Function to train a model with a specific aggregation strategy
def train_with_strategy(strategy, weights=None, output_dir="./results"):
    print(f"\n\nTraining with strategy: {strategy}")
    if weights is not None:
        print(f"Using custom weights: {weights}")
    
    # Load a fresh copy of the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Create a reward model
    reward_model = CombinedRewardModel(
        process_model_name=model_name,
        outcome_model_name=model_name,
        process_weight=0.7,
        outcome_weight=0.3,
        device=device
    )
    
    # Configure the Stepwise DPO trainer
    strategy_dir = f"{strategy}_custom" if weights is not None else strategy
    training_args = StepwiseDPOConfig(
        output_dir=f"{output_dir}/{strategy_dir}",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        report_to="none",
        remove_unused_columns=False,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        max_steps=100,  # Small number for demonstration
        # Stepwise DPO specific parameters
        step_beta=0.1,
        outcome_beta=0.1,
        aggregation_strategy=strategy,
        step_weights=weights,
        step_delimiter="\n\nStep",
        step_loss_weight=0.7,
        outcome_loss_weight=0.3,
        evaluate_step_accuracy=True,
    )
    
    # Initialize the trainer
    trainer = StepwiseDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        reward_model=reward_model,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(f"{output_dir}/{strategy_dir}_final")
    
    return model, trainer

# Train models with different strategies
results = {}
for strategy in strategies:
    if strategy == "weighted_sum":
        # Train with uniform weights
        model_uniform, trainer_uniform = train_with_strategy(strategy)
        results[f"{strategy}_uniform"] = model_uniform
        
        # Train with custom weights
        model_custom, trainer_custom = train_with_strategy(strategy, custom_weights)
        results[f"{strategy}_custom"] = model_custom
    else:
        model, trainer = train_with_strategy(strategy)
        results[strategy] = model

# Evaluate models on a test prompt
test_prompt = "Solve the equation: 2x - 5 = 11"

# Function to generate a response and analyze steps
def evaluate_model(model, strategy_name):
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract steps
    steps = extract_steps(response, "\n\nStep")
    
    # Create a simple reward model for evaluation
    eval_reward_model = CombinedRewardModel(
        process_model_name=model_name,
        outcome_model_name=model_name,
        device=device
    )
    
    # Get rewards for each step
    step_rewards = [eval_reward_model.process_reward_model.get_reward_for_text(step) for step in steps]
    
    # Aggregate rewards using different strategies for comparison
    aggregated_rewards = {}
    for agg_strategy in strategies:
        if agg_strategy == "weighted_sum":
            # Try both uniform and custom weights
            aggregated_rewards[f"{agg_strategy}_uniform"] = aggregate_rewards(step_rewards, agg_strategy)
            aggregated_rewards[f"{agg_strategy}_custom"] = aggregate_rewards(step_rewards, agg_strategy, custom_weights)
        else:
            aggregated_rewards[agg_strategy] = aggregate_rewards(step_rewards, agg_strategy)
    
    return {
        "response": response,
        "steps": steps,
        "step_rewards": step_rewards,
        "aggregated_rewards": aggregated_rewards
    }

# Evaluate all models
evaluation_results = {}
for strategy_name, model in results.items():
    print(f"\nEvaluating model trained with {strategy_name}")
    evaluation_results[strategy_name] = evaluate_model(model, strategy_name)

# Print results
print("\n\nEvaluation Results:")
print(f"Test Prompt: {test_prompt}")

for strategy_name, eval_result in evaluation_results.items():
    print(f"\n\nModel trained with {strategy_name}:")
    print(f"Response:\n{eval_result['response']}")
    
    print("\nStep Rewards:")
    for i, reward in enumerate(eval_result['step_rewards']):
        print(f"Step {i+1}: {reward:.4f}")
    
    print("\nAggregated Rewards:")
    for agg_strategy, reward in eval_result['aggregated_rewards'].items():
        print(f"{agg_strategy}: {reward:.4f}")

# Visualize results
plt.figure(figsize=(12, 8))

# Plot average step reward for each model
avg_step_rewards = [np.mean(eval_result['step_rewards']) for eval_result in evaluation_results.values()]
plt.subplot(2, 1, 1)
plt.bar(evaluation_results.keys(), avg_step_rewards)
plt.title('Average Step Reward by Training Strategy')
plt.xlabel('Training Strategy')
plt.ylabel('Average Step Reward')
plt.xticks(rotation=45)

# Plot number of steps generated by each model
step_counts = [len(eval_result['steps']) for eval_result in evaluation_results.values()]
plt.subplot(2, 1, 2)
plt.bar(evaluation_results.keys(), step_counts)
plt.title('Number of Steps Generated by Training Strategy')
plt.xlabel('Training Strategy')
plt.ylabel('Number of Steps')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "aggregation_comparison.png"))
plt.close()

print("\nResults visualization saved to experiments/aggregation_comparison.png")

# Save detailed results to a JSON file
detailed_results = {}
for strategy_name, eval_result in evaluation_results.items():
    detailed_results[strategy_name] = {
        "response": eval_result['response'],
        "step_count": len(eval_result['steps']),
        "step_rewards": [float(reward) for reward in eval_result['step_rewards']],
        "aggregated_rewards": {k: float(v) for k, v in eval_result['aggregated_rewards'].items()}
    }

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "aggregation_comparison_results.json"), "w") as f:
    json.dump(detailed_results, f, indent=2)

print("Detailed results saved to experiments/aggregation_comparison_results.json")