import os
import sys
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add root path for custom trainer and reward models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer import StepwiseDPOTrainer, StepwiseDPOConfig
from reward_model import CombinedRewardModel, ProcessRewardModel, OutcomeRewardModel

# For LoRA setup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_math_dpo_data.json")
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset = Dataset.from_dict({
    "prompt": [item["prompt"] for item in raw_data["data"]],
    "chosen": [item["chosen"] for item in raw_data["data"]],
    "rejected": [item["rejected"] for item in raw_data["data"]],
})
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Eval dataset size: {len(dataset['test'])}")

# Load tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model with 8-bit quantization for LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# Prepare model for 8bit training
model = prepare_model_for_kbit_training(model)

# Add LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # adjust as per model
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Set up reward models (same base)
process_model = ProcessRewardModel(model_name_or_path=model_name, device=device)
outcome_model = OutcomeRewardModel(model_name_or_path=model_name, device=device)
reward_model = CombinedRewardModel(
    process_model=process_model,
    outcome_model=outcome_model,
    process_weight=0.7,
    outcome_weight=0.3,
    step_delimiter="Step"
)

# Trainer configuration
training_args = StepwiseDPOConfig(
    output_dir="./results/stepwise_dpo_lora",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=2e-5,
    report_to="wandb",
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

# Initialize StepwiseDPOTrainer
trainer = StepwiseDPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    reward_model=reward_model,
)

# Start training
trainer.train()

# Save LoRA-adapted model
trainer.save_model("./results/stepwise_dpo_lora_final")
print("Training complete!")

# Generate a sample output
prompt = "Solve the equation: 2x - 5 = 11"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    inputs.input_ids,
    max_length=1024,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nSample generation:")
print(f"Prompt: {prompt}")
print(f"Response:\n{response}")
