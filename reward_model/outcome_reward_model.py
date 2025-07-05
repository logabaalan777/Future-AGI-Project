import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer


class OutcomeRewardModel(nn.Module):
    """Reward model for evaluating the quality of final outcomes.
    
    This model takes a complete reasoning process and final answer as input
    and outputs a scalar reward indicating the quality of the outcome.
    It can be used to provide outcome-level rewards for the StepwiseDPOTrainer.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[str] = None,
        max_length: int = 2048,
        use_answer_prompt: bool = True,
        answer_prompt: str = "Is the final answer correct? Yes or No:",
    ):
        """Initialize the OutcomeRewardModel.
        
        Args:
            model_name_or_path: The name or path of the base model to use
            tokenizer: The tokenizer for the model
            device: The device to use for inference
            max_length: The maximum length of input sequences
            use_answer_prompt: Whether to append an answer prompt to the outcome
            answer_prompt: The prompt to use for answer verification
        """
        super().__init__()
        
        # Load the base model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.processing_class = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Set the device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set the maximum length
        self.max_length = max_length
        
        # Set the answer prompt
        self.use_answer_prompt = use_answer_prompt
        self.answer_prompt = answer_prompt
        
        # Add a reward head on top of the model
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1)
        
        # Initialize the reward head
        self.reward_head.weight.data.normal_(mean=0.0, std=0.02)
        self.reward_head.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.FloatTensor]:
        """Forward pass of the OutcomeRewardModel.
        
        Args:
            input_ids: The input token IDs
            attention_mask: The attention mask
            return_dict: Whether to return a dictionary
            
        Returns:
            A dictionary containing the rewards
        """
        # Get the model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        
        # Get the rewards from the reward head
        rewards = self.reward_head(last_hidden_state).squeeze(-1)
        
        # Get the rewards for the last token in each sequence
        if attention_mask is not None:
            # Get the position of the last token in each sequence
            last_token_pos = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            rewards = rewards[torch.arange(batch_size), last_token_pos]
        else:
            # If no attention mask, just take the last token
            rewards = rewards[:, -1]
        
        if return_dict:
            return {"rewards": rewards}
        return rewards
    
    def get_reward(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 4,
    ) -> torch.FloatTensor:
        """Get rewards for a batch of texts.
        
        Args:
            texts: The texts to get rewards for
            batch_size: The batch size to use for inference
            
        Returns:
            A tensor of rewards
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Add answer prompt if needed
        if self.use_answer_prompt:
            texts = [f"{text}\n{self.answer_prompt}" for text in texts]
        
        # Initialize rewards tensor
        rewards = torch.zeros(len(texts), device=self.device)
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize the batch
            inputs = self.processing_class(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            
            # Get rewards
            with torch.no_grad():
                batch_rewards = self.forward(**inputs, return_dict=False)
            
            # Store rewards
            rewards[i:i+batch_size] = batch_rewards
        
        return rewards
    
    def get_outcome_rewards(
        self,
        problems: List[str],
        solutions: List[str],
        batch_size: int = 4,
    ) -> Dict[str, torch.FloatTensor]:
        """Get rewards for problem-solution pairs.
        
        Args:
            problems: The problem statements
            solutions: The solutions to evaluate
            batch_size: The batch size to use for inference
            
        Returns:
            A dictionary containing the rewards and other metrics
        """
        # Combine problems and solutions
        texts = [f"Problem: {problem}\nSolution: {solution}" for problem, solution in zip(problems, solutions)]
        
        # Get rewards for each outcome
        rewards = self.get_reward(texts, batch_size=batch_size)
        
        # Compute additional metrics
        mean_reward = rewards.mean().item()
        min_reward = rewards.min().item()
        max_reward = rewards.max().item()
        std_reward = rewards.std().item()
        
        return {
            "rewards": rewards,
            "mean_reward": mean_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "std_reward": std_reward,
        }
    
    def save_pretrained(self, save_directory: str):
        """Save the model and tokenizer to a directory.
        
        Args:
            save_directory: The directory to save to
        """
        # Create the directory if it doesn't exist
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model
        self.model.save_pretrained(os.path.join(save_directory, "base_model"))
        
        # Save the tokenizer
        self.processing_class.save_pretrained(os.path.join(save_directory, "tokenizer"))
        
        # Save the reward head
        torch.save(self.reward_head.state_dict(), os.path.join(save_directory, "reward_head.pt"))
        
        # Save the configuration
        import json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(
                {
                    "max_length": self.max_length,
                    "use_answer_prompt": self.use_answer_prompt,
                    "answer_prompt": self.answer_prompt,
                },
                f,
                indent=2,
            )
    
    @classmethod
    def from_pretrained(cls, load_directory: str, device: Optional[str] = None):
        """Load the model and tokenizer from a directory.
        
        Args:
            load_directory: The directory to load from
            device: The device to use for inference
            
        Returns:
            The loaded OutcomeRewardModel
        """
        # Load the configuration
        import os
        import json
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        
        # Load the base model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(os.path.join(load_directory, "base_model"))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "tokenizer"))
        
        # Create the reward model
        reward_model = cls(
            model_name_or_path=os.path.join(load_directory, "base_model"),
            tokenizer=tokenizer,
            device=device,
            max_length=config["max_length"],
            use_answer_prompt=config["use_answer_prompt"],
            answer_prompt=config["answer_prompt"],
        )
        
        # Load the reward head
        reward_model.reward_head.load_state_dict(
            torch.load(os.path.join(load_directory, "reward_head.pt"), map_location=reward_model.device)
        )
        
        return reward_model