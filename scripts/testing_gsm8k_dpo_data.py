import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reward_model.combined_reward_model import CombinedRewardModel
from trainer.utils import extract_steps_from_string, aggregate_rewards
from reward_model.process_reward_model import ProcessRewardModel
from reward_model.outcome_reward_model import OutcomeRewardModel

# Configuration
MODEL_DIR = "./results/stepwise_dpo_gsm8k_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEP_DELIMITER = "Step"  # Avoid using '\n\nStep'

# Load tokenizer and set pad token
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)
model.eval()

# Load combined reward model
process_model = ProcessRewardModel(model_name_or_path=MODEL_DIR, device=DEVICE)
outcome_model = OutcomeRewardModel(model_name_or_path=MODEL_DIR, device=DEVICE)
reward_model = CombinedRewardModel(
    process_model=process_model,
    outcome_model=outcome_model,
    process_weight=0.7,
    outcome_weight=0.3
)
print("✅ Fully trained model and reward model loaded successfully!")

# Prompt
prompt = (
    "Let's solve this step by step.\n"
    "A store has 150 items in stock. If they sell 30% on Monday and 40% of the rest on Tuesday, how many remain?"
)

# Generate model response
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== Sample Generation ===")
print(f"Prompt:\n{prompt}")
print(f"\nModel Response:\n{response}")

# Extract steps using robust splitting (fallback if no "Step" is found)
steps = extract_steps_from_string(response, step_delimiter=STEP_DELIMITER)
if not steps:
    print("\n⚠️ No 'Step' delimiters found; trying generic split...")
    steps = [s.strip() for s in response.split('\n') if s.strip()]

print("\n=== Extracted Steps ===")
for i, step in enumerate(steps):
    print(f"Step {i+1}: {step}")

# Get reward for each step using ProcessRewardModel only
step_scores = []
for i, step in enumerate(steps):
    try:
        score = reward_model.process_model.get_reward(step)
    except Exception as e:
        print(f"⚠️ Error scoring step {i+1}: {e}")
        score = 0.0
    step_scores.append(score)

print("\n=== Step Rewards ===")
for i, score in enumerate(step_scores):
    print(f"Step {i+1} Reward: {score.item():.4f}" if isinstance(score, torch.Tensor) else f"Step {i+1} Reward: {score:.4f}")

# Aggregate step rewards
step_scores_tensor = torch.tensor([step_scores], dtype=torch.float32).to(DEVICE)
final_score = aggregate_rewards(step_scores_tensor, strategy="weighted_sum")

print(f"\n🎯 Aggregated Reward Score: {final_score.item():.4f}")