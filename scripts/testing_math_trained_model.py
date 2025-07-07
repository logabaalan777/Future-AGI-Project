from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# === Configuration ===
peft_model_path = "./results/stepwise_dpo_lora_final"
MAX_CONTEXT_LENGTH = 2048  # Set according to your base model
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.95

# === Load PEFT config and base model ===
peft_config = PeftConfig.from_pretrained(peft_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Set pad/eos token if missing
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id or tokenizer.unk_token_id
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = tokenizer.pad_token_id

# === Load LoRA adapter ===
model = PeftModel.from_pretrained(base_model, peft_model_path)
model.eval()

# === Generation Function ===
def generate_response(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Calculate max tokens safely
    max_tokens = MAX_CONTEXT_LENGTH - input_ids.shape[-1]
    if max_tokens <= 0:
        raise ValueError("Prompt too long for model context window.")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# === Run test prompt ===
prompt = "Solve the equation: 2x - 5 = 11"
response = generate_response(prompt)
print("Generated Response:\n", response)
