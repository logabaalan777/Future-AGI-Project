import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.utils import extract_steps, extract_steps_from_string
from reward_model import CombinedRewardModel, ProcessRewardModel, OutcomeRewardModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to the trained model
model_path = "./results/stepwise_dpo_lora_final"

# Check if model path exists
if not os.path.exists(model_path):
    print(f"Model path {model_path} does not exist. Please run train_example.py first.")
    print("Using the original model instead...")
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Use original model if trained model doesn't exist

# Load the model and tokenizer
print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load sample data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_math_dpo_data.json")
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Create a simple reward model (or load a trained one if available)
reward_model_path = "./results/reward_model"
if os.path.exists(reward_model_path):
    reward_model = CombinedRewardModel.from_pretrained(reward_model_path, device=device)
    print(f"Loaded reward model from {reward_model_path}")
else:
    # Create a simple reward model using the same model
    process_model = ProcessRewardModel(
        model_name_or_path=model_path,
        device=device
    )
    outcome_model = OutcomeRewardModel(
        model_name_or_path=model_path,
        device=device
    )
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
    
    # Remove the prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):]
    
    return response.strip()

# Function to extract steps and compute rewards
def compute_step_rewards(response, step_delimiter="\n\nStep"):
    # Extract steps
    steps = extract_steps_from_string(response, step_delimiter)
    
    if not steps:
        return [], []
    
    # Get rewards for each step
    step_rewards = [reward_model.get_reward(step)["combined_reward"] for step in steps]
    
    return steps, step_rewards

# Function to visualize step rewards
def visualize_step_rewards(steps, rewards, title):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create step labels (Step 1, Step 2, etc.)
    step_labels = [f"Step {i+1}" for i in range(len(steps))]
    
    # Create the bar chart
    bars = ax.bar(step_labels, rewards, color='skyblue')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    
    # Add the reward values on top of the bars
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{reward:.2f}', ha='center', va='bottom')
    
    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Function to visualize step rewards for multiple responses
def visualize_multiple_responses(prompts, responses, chosen_responses, rejected_responses):
    # Create a figure with subplots
    fig, axes = plt.subplots(len(prompts), 3, figsize=(15, 5 * len(prompts)))
    
    for i, (prompt, response, chosen, rejected) in enumerate(zip(prompts, responses, chosen_responses, rejected_responses)):
        # Compute rewards for generated response
        gen_steps, gen_rewards = compute_step_rewards(response)
        
        # Compute rewards for chosen response
        chosen_steps, chosen_rewards = compute_step_rewards(chosen)
        
        # Compute rewards for rejected response
        rejected_steps, rejected_rewards = compute_step_rewards(rejected)
        
        # Plot rewards for generated response
        if len(gen_steps) > 0:
            ax = axes[i, 0] if len(prompts) > 1 else axes[0]
            bars = ax.bar([f"Step {j+1}" for j in range(len(gen_steps))], gen_rewards, color='skyblue')
            ax.set_title(f"Generated Response\nPrompt: {prompt[:30]}...")
            ax.set_ylabel('Reward')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            for bar, reward in zip(bars, gen_rewards):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{reward:.2f}', ha='center', va='bottom')
        
        # Plot rewards for chosen response
        if len(chosen_steps) > 0:
            ax = axes[i, 1] if len(prompts) > 1 else axes[1]
            bars = ax.bar([f"Step {j+1}" for j in range(len(chosen_steps))], chosen_rewards, color='green')
            ax.set_title("Chosen Response (Reference)")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            for bar, reward in zip(bars, chosen_rewards):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{reward:.2f}', ha='center', va='bottom')
        
        # Plot rewards for rejected response
        if len(rejected_steps) > 0:
            ax = axes[i, 2] if len(prompts) > 1 else axes[2]
            bars = ax.bar([f"Step {j+1}" for j in range(len(rejected_steps))], rejected_rewards, color='salmon')
            ax.set_title("Rejected Response (Reference)")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            for bar, reward in zip(bars, rejected_rewards):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{reward:.2f}', ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Create output directory for visualizations
output_dir = "./results/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Test the model on sample prompts and visualize the results
print("\nGenerating responses and visualizing step rewards...\n")

# Collect data for visualization
prompts = []
responses = []
chosen_responses = []
rejected_responses = []

for i, item in enumerate(raw_data["data"]):
    prompt = item["prompt"]
    chosen = item["chosen"]
    rejected = item["rejected"]
    
    print(f"Processing prompt {i+1}: {prompt}")
    
    # Generate a response
    response = generate_response(prompt)
    
    # Extract steps and compute rewards
    steps, rewards = compute_step_rewards(response)
    
    if not steps:
        print(f"No steps found in the response for prompt {i+1}")
        continue
    
    # Visualize individual response
    fig = visualize_step_rewards(steps, rewards, f"Step Rewards for Prompt {i+1}")
    fig.savefig(os.path.join(output_dir, f"prompt_{i+1}_rewards.png"))
    plt.close(fig)
    
    # Add to collection for combined visualization
    prompts.append(prompt)
    responses.append(response)
    chosen_responses.append(chosen)
    rejected_responses.append(rejected)

# Visualize all responses together
if prompts:
    fig = visualize_multiple_responses(prompts, responses, chosen_responses, rejected_responses)
    fig.savefig(os.path.join(output_dir, "all_responses_comparison.png"))
    plt.close(fig)

print(f"\nVisualizations saved to {output_dir}")
print("\nTo view the visualizations, check the PNG files in the results/visualizations directory.")