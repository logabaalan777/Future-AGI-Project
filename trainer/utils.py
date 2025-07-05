import re
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from transformers import PreTrainedModel, PreTrainedTokenizer


def extract_steps(
    input_ids: torch.LongTensor,
    tokenizer: PreTrainedTokenizer,
    step_delimiter: str = "Step",
    max_steps: int = 8,
) -> List[torch.LongTensor]:
    """Extract individual reasoning steps from model outputs.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: Tokenizer for decoding
        step_delimiter: String used to identify steps
        max_steps: Maximum number of steps to extract
        
    Returns:
        List of token IDs for each step [batch_size, num_steps, step_seq_len]
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    # Convert input_ids to text
    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    
    # Initialize list to store step token IDs
    all_step_ids = [[] for _ in range(batch_size)]
    
    # Extract steps from each text in the batch
    for i, text in enumerate(texts):
        # Use regex to find steps based on delimiter
        step_pattern = f"{step_delimiter}\s*\d+:?\s*(.+?)(?={step_delimiter}\s*\d+:|$)"
        steps = re.findall(step_pattern, text, re.DOTALL)
        
        # Limit to max_steps
        steps = steps[:max_steps]
        
        # Tokenize each step
        for step in steps:
            step_ids = tokenizer(step.strip(), return_tensors="pt").input_ids.to(device)
            all_step_ids[i].append(step_ids)
    
    return all_step_ids

# def extract_steps_from_string(
#     text: str,
#     step_delimiter: str = "\n\nStep",
#     max_steps: int = 8
# ) -> List[str]:
#     """
#     Extract individual steps from a response string.

#     Args:
#         text: The generated text from the model
#         step_delimiter: Delimiter used to identify steps
#         max_steps: Max steps to extract

#     Returns:
#         List of step strings
#     """
#     pattern = f"{re.escape(step_delimiter)}\s*\d+:?\s*(.+?)(?={re.escape(step_delimiter)}\s*\d+:|$)"
#     steps = re.findall(pattern, text.strip(), re.DOTALL)
#     return [step.strip() for step in steps[:max_steps]]

def extract_steps_from_string(
    text: str,
    step_delimiter: str = "\n\nStep",
    max_steps: int = 8,
) -> List[str]:
    """
    Extract individual steps from a response string.

    Args:
        text: The generated text from the model
        step_delimiter: Delimiter used to identify steps
        max_steps: Max steps to extract

    Returns:
        List of step strings
    """
    step_pattern = r"(?:Step\s*\d+[:.)]?|^\d+[).])\s*(.+?)(?=(?:Step\s*\d+[:.)]?|^\d+[).])|$)"
    matches = re.findall(step_pattern, text.strip(), re.DOTALL | re.MULTILINE)
    return [step.strip() for step in matches[:max_steps]]


def get_step_rewards(
    model: PreTrainedModel,
    step_ids: List[List[torch.LongTensor]],
    tokenizer: PreTrainedTokenizer,
    strategy: str = "weighted_sum",
    weights: Optional[List[float]] = None,
    external_reward_model: Optional[PreTrainedModel] = None,
) -> torch.FloatTensor:
    """Compute rewards for individual reasoning steps."""
    batch_size = len(step_ids)
    device = model.device

    reward_model = external_reward_model or model

    # Initialize rewards tensor
    max_steps = max(len(steps) for steps in step_ids)
    rewards = torch.zeros((batch_size, max_steps), device=device)

    # Compute rewards
    for i, example_steps in enumerate(step_ids):
        for j, step in enumerate(example_steps):
            # Ensure step tensor is on correct device
            step = step.to(device)

            # Verification prompt on same device
            verification_prompt = tokenizer("Is this step correct? Yes or No:", return_tensors="pt").input_ids.to(device)

            # Concatenate step with prompt
            input_ids = torch.cat([step.squeeze(0), verification_prompt.squeeze(0)], dim=0).unsqueeze(0).to(device)

            # Run through model
            with torch.no_grad():
                outputs = reward_model(input_ids)
                logits = outputs.logits

            # Softmax on last token
            last_token_logits = logits[0, -1]
            yes_token_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
            no_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]

            probs = F.softmax(last_token_logits, dim=0)
            yes_prob = probs[yes_token_id]
            no_prob = probs[no_token_id]

            # Log-ratio as reward
            reward = torch.log(yes_prob / (no_prob + 1e-10))
            rewards[i, j] = reward

    return aggregate_rewards(rewards, strategy, weights)


def aggregate_rewards(
    rewards: torch.FloatTensor,
    strategy: str = "weighted_sum",
    weights: Optional[List[float]] = None,
) -> torch.FloatTensor:
    """Aggregate step-level rewards using different strategies.
    
    Args:
        rewards: Tensor of rewards for each step [batch_size, num_steps]
        strategy: Strategy for aggregating rewards
        weights: Optional weights for each step in weighted_sum strategy
        
    Returns:
        Tensor of aggregated rewards [batch_size]
    """
    batch_size, num_steps = rewards.shape
    device = rewards.device
    
    # Create a mask for valid (non-zero) rewards
    mask = (rewards != 0).float()
    
    # Apply different aggregation strategies
    if strategy == "weighted_sum":
        if weights is None:
            # Default to equal weights
            weights = torch.ones(num_steps, device=device) / num_steps
        else:
            # Convert weights to tensor and normalize
            weights = torch.tensor(weights, device=device)
            weights = weights / weights.sum()
            
            # Pad or truncate weights to match num_steps
            if len(weights) < num_steps:
                weights = torch.cat([weights, torch.zeros(num_steps - len(weights), device=device)])
            elif len(weights) > num_steps:
                weights = weights[:num_steps]
        
        # Apply weights and mask
        weighted_rewards = rewards * weights.unsqueeze(0) * mask
        aggregated = weighted_rewards.sum(dim=1)
    
    elif strategy == "min_step":
        # Replace zeros with large positive value before taking min
        masked_rewards = rewards.clone()
        masked_rewards[mask == 0] = 1e10
        aggregated = masked_rewards.min(dim=1).values
        
        # If all steps are invalid (all zeros), return zero
        all_invalid = (mask.sum(dim=1) == 0)
        aggregated[all_invalid] = 0
    
    elif strategy == "harmonic_mean":
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Count valid steps
        valid_steps = mask.sum(dim=1)
        
        # Compute harmonic mean of valid rewards
        harmonic_sum = (mask / (rewards.abs() + epsilon)).sum(dim=1)
        aggregated = valid_steps / (harmonic_sum + epsilon)
        
        # Preserve sign of the average reward
        sign = torch.sign((rewards * mask).sum(dim=1))
        aggregated = aggregated * sign
        
        # If all steps are invalid, return zero
        aggregated[valid_steps == 0] = 0
    
    elif strategy == "geometric_mean":
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        
        # Count valid steps
        valid_steps = mask.sum(dim=1)
        
        # Compute log of absolute rewards
        log_abs_rewards = torch.log(rewards.abs() + epsilon) * mask
        
        # Compute geometric mean of valid rewards
        log_geo_mean = log_abs_rewards.sum(dim=1) / (valid_steps + epsilon)
        aggregated = torch.exp(log_geo_mean)
        
        # Preserve sign of the average reward
        sign = torch.sign((rewards * mask).sum(dim=1))
        aggregated = aggregated * sign
        
        # If all steps are invalid, return zero
        aggregated[valid_steps == 0] = 0
    
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    
    return aggregated


def compute_step_loss(
    chosen_steps_logps: torch.FloatTensor,
    rejected_steps_logps: torch.FloatTensor,
    step_rewards_chosen: torch.FloatTensor,
    step_rewards_rejected: torch.FloatTensor,
    beta: float = 0.1,
) -> torch.FloatTensor:
    """Compute the step-level DPO loss.
    
    Args:
        chosen_steps_logps: Log probabilities for chosen steps
        rejected_steps_logps: Log probabilities for rejected steps
        step_rewards_chosen: Rewards for chosen steps
        step_rewards_rejected: Rewards for rejected steps
        beta: Temperature parameter for the DPO loss
        
    Returns:
        The step-level DPO loss
    """
    # Compute the advantage (difference in rewards)
    advantage = step_rewards_chosen - step_rewards_rejected
    
    # Compute the log ratios between chosen and rejected
    log_ratio = chosen_steps_logps - rejected_steps_logps
    
    # Compute the DPO loss with the advantage as a weight
    logits = beta * log_ratio
    loss = -F.logsigmoid(logits) * advantage.abs()
    
    # Take the mean over the batch
    loss = loss.mean()
    
    return loss