import os
import sys
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import StepwiseDPOTrainer, StepwiseDPOConfig
from reward_model import CombinedRewardModel, ProcessRewardModel, OutcomeRewardModel

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset (JSONL format)
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_gsm8k_dpo_data.jsonl")
dataset = load_dataset("json", data_files=data_path)

# Split dataset
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
print(f"Train size: {len(dataset['train'])}, Eval size: {len(dataset['test'])}")

# Model name (LoRA compatible)
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # Adjust for your model
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Reward model
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

# Training configuration
training_args = StepwiseDPOConfig(
    output_dir="./results/stepwise_dpo_gsm8k",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=2e-5,
    report_to="wandb",  # Set to 'wandb' if using Weights & Biases
    remove_unused_columns=False,
    optim="adamw_torch",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    max_steps=100,
    step_beta=0.1,
    outcome_beta=0.1,
    aggregation_strategy="weighted_sum",
    step_weights=None,
    step_delimiter="\n\nStep",
    step_loss_weight=0.7,
    max_grad_norm=1.0,
    outcome_loss_weight=0.3,
    evaluate_step_accuracy=True,
)

# Trainer
trainer = StepwiseDPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    reward_model=reward_model,
)

# Train
trainer.train()

# Save final model and LoRA adapter
trainer.save_model("./results/stepwise_dpo_gsm8k_final")
model.save_pretrained("./results/lora_adapter")
print("✅ Training complete and model saved!")

# Generate sample output
prompt = "A store has 150 items in stock. If they sell 30% on Monday and 40% of the rest on Tuesday, how many remain?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    inputs.input_ids,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nSample Generation:")
print(f"Prompt: {prompt}")
print(f"Response:\n{response}")

# Evaluate generated steps
from trainer import extract_steps, aggregate_rewards

steps = extract_steps(response, training_args.step_delimiter)
print("\nExtracted Steps:")
for i, step in enumerate(steps):
    print(f"Step {i+1}: {step}")

step_rewards = [reward_model.process_model.get_reward(step) for step in steps]
print("\nStep Rewards:")
for i, reward in enumerate(step_rewards):
    print(f"Step {i+1} Reward: {reward:.4f}")

final_score = aggregate_rewards(step_rewards, strategy="weighted_sum")
print(f"\nAggregated Reward Score: {final_score:.4f}")
