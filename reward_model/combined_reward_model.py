import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Dict, List, Optional, Tuple, Union, Any

from transformers import PreTrainedTokenizer

from .process_reward_model import ProcessRewardModel
from .outcome_reward_model import OutcomeRewardModel

class CombinedRewardModel(nn.Module):
    """Combined reward model that integrates process and outcome rewards.
    
    This model combines a ProcessRewardModel for step-level rewards and an
    OutcomeRewardModel for outcome-level rewards, providing a comprehensive
    evaluation of both the reasoning process and the final answer.
    """
    
    def __init__(
        self,
        process_model: ProcessRewardModel,
        outcome_model: OutcomeRewardModel,
        process_weight: float = 0.5,
        outcome_weight: float = 0.5,
        step_delimiter: str = "Step",
        max_steps: int = 8,
        aggregation_strategy: str = "weighted_sum",
        step_weights: Optional[List[float]] = None,
    ):
        """Initialize the CombinedRewardModel.
        
        Args:
            process_model: The ProcessRewardModel for step-level rewards
            outcome_model: The OutcomeRewardModel for outcome-level rewards
            process_weight: The weight for process rewards
            outcome_weight: The weight for outcome rewards
            step_delimiter: String used to identify steps
            max_steps: Maximum number of steps to consider
            aggregation_strategy: Strategy for aggregating step rewards
            step_weights: Optional weights for each step in weighted_sum strategy
        """
        super().__init__()
        
        # Store the models
        self.process_model = process_model
        self.outcome_model = outcome_model
        
        # Ensure the models are on the same device
        self.device = self.process_model.device
        self.outcome_model.to(self.device)
        
        # Store the weights
        self.process_weight = process_weight
        self.outcome_weight = outcome_weight
        
        # Store the step parameters
        self.step_delimiter = step_delimiter
        self.max_steps = max_steps
        self.aggregation_strategy = aggregation_strategy
        self.step_weights = step_weights
        
        # Use the tokenizer from the process model
        self.processing_class = None
    
    def extract_steps(self, text: str) -> List[str]:
        """Extract individual reasoning steps from text.
        
        Args:
            text: The text to extract steps from
            
        Returns:
            A list of extracted steps
        """
        # Use regex to find steps based on delimiter
        step_pattern = f"{self.step_delimiter}\s*\d+:?\s*(.+?)(?={self.step_delimiter}\s*\d+:|$)"
        steps = re.findall(step_pattern, text, re.DOTALL)
        
        # Limit to max_steps
        steps = [step.strip() for step in steps[:self.max_steps]]
        
        return steps
    
    def aggregate_step_rewards(self, step_rewards: torch.FloatTensor) -> torch.FloatTensor:
        """Aggregate step rewards using the specified strategy.
        
        Args:
            step_rewards: Tensor of rewards for each step [num_steps]
            
        Returns:
            The aggregated reward
        """
        # Convert to tensor if not already
        if not isinstance(step_rewards, torch.Tensor):
            step_rewards = torch.tensor(step_rewards, device=self.device)
        
        # Apply different aggregation strategies
        if self.aggregation_strategy == "weighted_sum":
            if self.step_weights is None:
                # Default to equal weights
                weights = torch.ones_like(step_rewards) / len(step_rewards)
            else:
                # Convert weights to tensor and normalize
                weights = torch.tensor(self.step_weights, device=self.device)
                weights = weights[:len(step_rewards)]  # Truncate if needed
                weights = weights / weights.sum()
            
            # Apply weights
            return (step_rewards * weights).sum()
        
        elif self.aggregation_strategy == "min_step":
            return step_rewards.min()
        
        elif self.aggregation_strategy == "harmonic_mean":
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            return len(step_rewards) / ((1 / (step_rewards.abs() + epsilon)).sum())
        
        elif self.aggregation_strategy == "geometric_mean":
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            return torch.exp(torch.log(step_rewards.abs() + epsilon).mean())
        
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
    
    def get_reward(
        self,
        text: str,
        problem: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get combined rewards for a text.
        
        Args:
            text: The text to get rewards for
            problem: Optional problem statement for outcome evaluation
            
        Returns:
            A dictionary containing the rewards and other metrics
        """
        # Extract steps from the text
        steps = self.extract_steps(text)
        
        # Get process rewards for each step
        if steps:
            step_rewards = self.process_model.get_reward(steps)
            process_reward = self.aggregate_step_rewards(step_rewards)
        else:
            step_rewards = torch.tensor([], device=self.device)
            process_reward = torch.tensor(0.0, device=self.device)
        
        # Get outcome reward
        if problem is not None:
            outcome_text = f"Problem: {problem}\nSolution: {text}"
        else:
            outcome_text = text
        
        outcome_reward = self.outcome_model.get_reward(outcome_text)
        
        # Combine rewards
        combined_reward = (
            self.process_weight * process_reward +
            self.outcome_weight * outcome_reward
        )
        
        return {
            "combined_reward": combined_reward.item(),
            "process_reward": process_reward.item(),
            "outcome_reward": outcome_reward.item(),
            "step_rewards": step_rewards.tolist() if len(step_rewards) > 0 else [],
            "num_steps": len(steps),
        }
    
    def batch_get_reward(
        self,
        texts: List[str],
        problems: Optional[List[str]] = None,
        batch_size: int = 4,
    ) -> List[Dict[str, Any]]:
        """Get combined rewards for a batch of texts.
        
        Args:
            texts: The texts to get rewards for
            problems: Optional problem statements for outcome evaluation
            batch_size: The batch size to use for inference
            
        Returns:
            A list of dictionaries containing the rewards and other metrics
        """
        # Initialize results
        results = []
        
        # Process each text individually
        for i, text in enumerate(texts):
            problem = problems[i] if problems is not None else None
            result = self.get_reward(text, problem)
            results.append(result)
        
        return results
    
    def save_pretrained(self, save_directory: str):
        """Save the models and configuration to a directory.
        
        Args:
            save_directory: The directory to save to
        """
        # Create the directory if it doesn't exist
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the process model
        self.process_model.save_pretrained(os.path.join(save_directory, "process_model"))
        
        # Save the outcome model
        self.outcome_model.save_pretrained(os.path.join(save_directory, "outcome_model"))
        
        # Save the configuration
        import json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(
                {
                    "process_weight": self.process_weight,
                    "outcome_weight": self.outcome_weight,
                    "step_delimiter": self.step_delimiter,
                    "max_steps": self.max_steps,
                    "aggregation_strategy": self.aggregation_strategy,
                    "step_weights": self.step_weights,
                },
                f,
                indent=2,
            )
    
    @classmethod
    def from_pretrained(cls, load_directory: str, device: Optional[str] = None):
        """Load the models and configuration from a directory.
        
        Args:
            load_directory: The directory to load from
            device: The device to use for inference
            
        Returns:
            The loaded CombinedRewardModel
        """
        # Load the configuration
        import os
        import json
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        
        # Load the process model
        process_model = ProcessRewardModel.from_pretrained(
            os.path.join(load_directory, "process_model"),
            device=device,
        )
        
        # Load the outcome model
        outcome_model = OutcomeRewardModel.from_pretrained(
            os.path.join(load_directory, "outcome_model"),
            device=device,
        )
        
        # Create the combined model
        combined_model = cls(
            process_model=process_model,
            outcome_model=outcome_model,
            process_weight=config["process_weight"],
            outcome_weight=config["outcome_weight"],
            step_delimiter=config["step_delimiter"],
            max_steps=config["max_steps"],
            aggregation_strategy=config["aggregation_strategy"],
            step_weights=config["step_weights"],
        )
        
        return combined_model