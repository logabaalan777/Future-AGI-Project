import os
import json
import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


class ProcessRewardModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        use_verification_prompt: bool = True,
        verification_prompt: str = "Is this step correct? Yes or No:",
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path)

        self.max_length = max_length
        self.use_verification_prompt = use_verification_prompt
        self.verification_prompt = verification_prompt

        self.reward_head = nn.Linear(self.model.config.hidden_size, 1).to(self.device)
        self.reward_head.weight.data.normal_(mean=0.0, std=0.02)
        self.reward_head.bias.data.zero_()

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.FloatTensor] = None, return_dict: bool = True):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        rewards = self.reward_head(last_hidden_state).squeeze(-1)  # [batch_size, seq_len]

        if attention_mask is not None:
            last_token_pos = attention_mask.sum(dim=1) - 1
            rewards = rewards[torch.arange(rewards.size(0), device=self.device), last_token_pos]
        else:
            rewards = rewards[:, -1]

        return {"rewards": rewards} if return_dict else rewards

    def get_reward(self, texts: Union[str, List[str]], batch_size: int = 8) -> torch.FloatTensor:
        if isinstance(texts, str):
            texts = [texts]

        if self.use_verification_prompt:
            texts = [f"{text}\n{self.verification_prompt}" for text in texts]

        rewards = torch.zeros(len(texts), device=self.device)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)

            with torch.no_grad():
                batch_rewards = self.forward(**inputs, return_dict=False)

            rewards[i:i+batch_size] = batch_rewards

        return rewards

    def get_step_rewards(self, steps: List[str], batch_size: int = 8) -> Dict[str, torch.FloatTensor]:
        rewards = self.get_reward(steps, batch_size=batch_size)
        return {
            "rewards": rewards,
            "mean_reward": rewards.mean().item(),
            "min_reward": rewards.min().item(),
            "max_reward": rewards.max().item(),
            "std_reward": rewards.std().item(),
        }

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(os.path.join(save_directory, "base_model"))
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        torch.save(self.reward_head.state_dict(), os.path.join(save_directory, "reward_head.pt"))

        config = {
            "max_length": self.max_length,
            "use_verification_prompt": self.use_verification_prompt,
            "verification_prompt": self.verification_prompt,
        }

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str, device: Optional[str] = None):
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "tokenizer"))
        instance = cls(
            model_name_or_path=os.path.join(load_directory, "base_model"),
            tokenizer=tokenizer,
            device=device,
            max_length=config["max_length"],
            use_verification_prompt=config["use_verification_prompt"],
            verification_prompt=config["verification_prompt"],
        )

        reward_head_path = os.path.join(load_directory, "reward_head.pt")
        instance.reward_head.load_state_dict(torch.load(reward_head_path, map_location=instance.device))
        instance.reward_head.to(instance.device)

        return instance
