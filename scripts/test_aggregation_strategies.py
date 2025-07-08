import os
import sys
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.utils import extract_steps, aggregate_rewards, extract_steps_from_string
from reward_model import CombinedRewardModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to the trained model
model_path = "./results/stepwise_dpo_lora_final"

# Check if model path exists
if not os.path.exists(model_path):
    print(f"Model path {model_path} does not exist. Please run train_example.py first.")
    print("Using the original model instead...")
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fallback model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Load main model
print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load sample data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_math_dpo_data.json")
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Create or load reward model
reward_model_path = "./results/reward_model"
if os.path.exists(reward_model_path):
    reward_model = CombinedRewardModel.load(reward_model_path, device=device)
    print(f"Loaded reward model from {reward_model_path}")
else:
    print("Creating reward model from process and outcome models...")

    process_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    outcome_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    reward_model = CombinedRewardModel(
        process_model=process_model,
        outcome_model=outcome_model,
        process_weight=0.7,
        outcome_weight=0.3
    )
    print("Created a simple reward model using the same model")

# Function to generate a response
def generate_response(prompt, max_length=512, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip() if response.startswith(prompt) else response.strip()

# Evaluate with a reward strategy
def evaluate_with_strategy(response, strategy, step_delimiter="\n\nStep", weights=None):
    steps = extract_steps_from_string(response, step_delimiter)

    if not steps:
        return 0.0, []

    step_rewards = [reward_model.get_reward_for_text(step) for step in steps]
    aggregated_reward = aggregate_rewards(step_rewards, strategy, weights)
    return aggregated_reward, step_rewards

# Reward aggregation strategies
strategies = [
    "weighted_sum",
    "min_step",
    "harmonic_mean",
    "geometric_mean",
]

# Custom weights
early_weights = [0.4, 0.3, 0.2, 0.1]
late_weights = [0.1, 0.2, 0.3, 0.4]

# Run evaluation on first 2 prompts
print("\nTesting different aggregation strategies...\n")
for i, item in enumerate(raw_data["data"][:2]):
    prompt = item["prompt"]
    print(f"Prompt {i + 1}: {prompt}")

    response = generate_response(prompt)
    print(f"\nGenerated response:\n{response}\n")

    steps = extract_steps_from_string(response, "\n\nStep")
    print(f"Number of steps extracted: {len(steps)}")

    print("\nEvaluation with different aggregation strategies:")
    for strategy in strategies:
        aggregated_reward, step_rewards = evaluate_with_strategy(response, strategy)
        print(f"\n{strategy.upper()}:")
        print(f"Step rewards: {[round(r, 3) for r in step_rewards]}")
        print(f"Aggregated reward: {round(aggregated_reward, 3)}")

    if len(steps) >= 4:
        print("\nWEIGHTED_SUM with custom weights:")

        weights = early_weights[:len(steps)]
        weights = [w / sum(weights) for w in weights]
        agg_reward, _ = evaluate_with_strategy(response, "weighted_sum", weights=weights)
        print(f"Early emphasis {weights}: {round(agg_reward, 3)}")

        weights = late_weights[:len(steps)]
        weights = [w / sum(weights) for w in weights]
        agg_reward, _ = evaluate_with_strategy(response, "weighted_sum", weights=weights)
        print(f"Late emphasis {weights}: {round(agg_reward, 3)}")

    print("-" * 80 + "\n")

print("Testing complete!")
