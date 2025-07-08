# Step Extraction and Verification in Stepwise DPO

This document explains how reasoning steps are extracted and verified in the Stepwise DPO trainer, based on the "Let's Verify Step by Step" methodology (arXiv:2408.15240v1).

## Overview

A key component of Stepwise DPO is the ability to extract individual reasoning steps from model outputs and evaluate them separately. This process involves:

1. **Step Extraction**: Parsing the model's output to identify distinct reasoning steps
2. **Step Verification**: Evaluating the correctness of each individual step
3. **Reward Calculation**: Converting verification results into reward signals

## Step Extraction

### Delimiter-Based Extraction

The most common approach for step extraction is to use a delimiter pattern to split the model's output into individual steps. The Stepwise DPO trainer implements this in the `extract_steps` function:

```python
def extract_steps(text, step_delimiter="\n\nStep", max_steps=None):
    """Extract individual reasoning steps from model output.
    
    Args:
        text (str): The model output text containing reasoning steps
        step_delimiter (str): The delimiter pattern used to identify steps
        max_steps (int, optional): Maximum number of steps to extract
        
    Returns:
        List[str]: List of extracted steps
    """
    # Implementation details...
```

### Common Delimiters

Several delimiter patterns can be used depending on the format of your training data:

- `"\n\nStep"`: For outputs formatted as "Step 1: ...", "Step 2: ...", etc.
- `"\n\n"`: For outputs with steps separated by blank lines
- `"\n[0-9]+\."`: For outputs with numbered steps like "1. ...", "2. ...", etc.

### Handling Edge Cases

The step extraction process handles several edge cases:

1. **Missing Delimiters**: If no delimiters are found, the entire text is treated as a single step
2. **Maximum Steps**: If `max_steps` is specified, only the first N steps are extracted
3. **Empty Steps**: Empty steps (after trimming whitespace) are filtered out

## Step Verification

Once steps are extracted, each step needs to be verified for correctness. The Stepwise DPO trainer implements this through reward models.

### Verification Approaches

#### 1. Binary Verification

The simplest approach is to use a binary verification model that classifies each step as correct or incorrect. This is implemented in the `get_step_rewards` function:

```python
def get_step_rewards(steps, model, tokenizer, device="cuda", verification_prompt="Is this step correct? Yes or No:"):
    """Compute rewards for individual reasoning steps.
    
    Args:
        steps (List[str]): List of reasoning steps to evaluate
        model: The model to use for verification
        tokenizer: The tokenizer for the model
        device (str): Device to run the model on
        verification_prompt (str): Prompt template for verification
        
    Returns:
        List[float]: List of rewards for each step
    """
    # Implementation details...
```

This function:
1. Prepends each step with a verification prompt
2. Computes the log probability ratio of "Yes" vs "No" responses
3. Returns this ratio as the reward for each step

#### 2. External Reward Model

For more sophisticated verification, an external reward model can be used. The `ProcessRewardModel` class implements this approach:

```python
class ProcessRewardModel(nn.Module):
    """Model for evaluating the quality of individual reasoning steps."""
    
    def __init__(self, model_name_or_path, device="cuda"):
        """Initialize the process reward model.
        
        Args:
            model_name_or_path (str): Path to the pretrained model
            device (str): Device to run the model on
        """
        # Implementation details...
        
    def get_reward_for_step(self, step):
        """Get reward for a single reasoning step.
        
        Args:
            step (str): The reasoning step to evaluate
            
        Returns:
            float: Reward value for the step
        """
        # Implementation details...
```

### Verification Prompts

The verification prompt plays a crucial role in step evaluation. Some effective prompt templates include:

- `"Is this step correct? Yes or No:"`: Simple binary verification
- `"Rate the correctness of this step on a scale from 1 to 10:"`: For more nuanced evaluation
- `"Identify any errors in this reasoning step:"`: For error-focused verification

## Reward Calculation

After verification, the results need to be converted into reward signals for the DPO training process.

### Log Probability Ratio

For binary verification, the reward is typically calculated as the log probability ratio:

```
reward = log(p("Yes") / p("No"))
```

This approach provides a continuous reward signal that reflects the model's confidence in the correctness of each step.

### Scaling and Normalization

Rewards may need to be scaled or normalized to ensure stable training:

```python
def scale_rewards(rewards, beta=0.1):
    """Scale rewards by beta parameter for DPO training.
    
    Args:
        rewards (List[float]): Raw rewards
        beta (float): Temperature parameter for DPO
        
    Returns:
        List[float]: Scaled rewards
    """
    return [reward * beta for reward in rewards]
```

## Integration with DPO Training

The extracted steps and their rewards are integrated into the DPO training process through the `compute_step_loss` function:

```python
def compute_step_loss(chosen_step_logps, rejected_step_logps, chosen_step_rewards, rejected_step_rewards, beta):
    """Compute the step-level DPO loss.
    
    Args:
        chosen_step_logps (torch.Tensor): Log probs for chosen steps
        rejected_step_logps (torch.Tensor): Log probs for rejected steps
        chosen_step_rewards (torch.Tensor): Rewards for chosen steps
        rejected_step_rewards (torch.Tensor): Rewards for rejected steps
        beta (float): Temperature parameter for DPO
        
    Returns:
        torch.Tensor: Step-level DPO loss
    """
    # Implementation details...
```

This function:
1. Calculates the log probability difference between chosen and rejected steps
2. Computes the advantage term based on reward differences
3. Combines these terms to produce the step-level DPO loss

## Data Preparation

To effectively use Stepwise DPO, training data should be prepared with clear step delimiters. The `prepare_data.py` script helps with this process:

```python
def add_step_markers(text, step_delimiter="\n\nStep"):
    """Add step markers to text if they are missing.
    
    Args:
        text (str): The text to process
        step_delimiter (str): The delimiter pattern to use
        
    Returns:
        str: Text with added step markers
    """
    # Implementation details...
```

## Best Practices

1. **Consistent Formatting**: Ensure that all training examples use consistent step delimiters
2. **Step Granularity**: Choose an appropriate level of granularity for steps (not too fine-grained, not too coarse)
3. **Verification Quality**: Use a high-quality verification model or prompt to ensure accurate step evaluation
4. **Balance**: Balance the weights between step-level and outcome-level losses

## References

1. "Let's Verify Step by Step" (arXiv:2408.15240v1)
2. Process Supervision: Going Beyond Step-by-Step Instructions (Lightman et al., 2023)
3. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)