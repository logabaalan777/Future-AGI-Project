# Using Custom Reward Models with Stepwise DPO

This document explains how to create and use custom reward models with the Stepwise DPO trainer, providing more sophisticated step evaluation than the default log probability approach.

## Overview

While Stepwise DPO can work with simple verification prompts, using dedicated reward models can significantly improve the quality of step evaluation. The framework supports three types of reward models:

1. **Process Reward Models**: Evaluate individual reasoning steps
2. **Outcome Reward Models**: Evaluate final outcomes
3. **Combined Reward Models**: Integrate both process and outcome evaluation

## Process Reward Models

Process reward models focus on evaluating the quality of individual reasoning steps.

### Implementation

The `ProcessRewardModel` class provides a foundation for step-level evaluation:

```python
class ProcessRewardModel(nn.Module):
    """Model for evaluating the quality of individual reasoning steps."""
    
    def __init__(self, model_name_or_path, device="cuda"):
        """Initialize the process reward model.
        
        Args:
            model_name_or_path (str): Path to the pretrained model
            device (str): Device to run the model on
        """
        super().__init__()
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Verification prompt template
        self.verification_prompt = "Is this step correct? Yes or No: "
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Model outputs
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
    def get_reward_for_text(self, text):
        """Get reward for a single text.
        
        Args:
            text (str): The text to evaluate
            
        Returns:
            float: Reward value for the text
        """
        # Prepare input with verification prompt
        input_text = self.verification_prompt + text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.forward(inputs.input_ids, inputs.attention_mask)
            logits = outputs.logits
        
        # Get log probabilities for "Yes" and "No"
        last_token_logits = logits[0, -1]
        yes_token_id = self.tokenizer.encode(" Yes", add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode(" No", add_special_tokens=False)[0]
        
        yes_logp = torch.log_softmax(last_token_logits, dim=0)[yes_token_id].item()
        no_logp = torch.log_softmax(last_token_logits, dim=0)[no_token_id].item()
        
        # Compute reward as log(p(Yes)/p(No))
        reward = yes_logp - no_logp
        
        return reward
    
    def get_rewards_for_steps(self, steps):
        """Get rewards for multiple steps.
        
        Args:
            steps (List[str]): List of reasoning steps to evaluate
            
        Returns:
            List[float]: List of rewards for each step
        """
        return [self.get_reward_for_text(step) for step in steps]
    
    def save(self, path):
        """Save the model and tokenizer.
        
        Args:
            path (str): Path to save the model
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save configuration
        config = {
            "verification_prompt": self.verification_prompt,
        }
        with open(os.path.join(path, "process_reward_config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path, device="cuda"):
        """Load the model from a path.
        
        Args:
            path (str): Path to load the model from
            device (str): Device to load the model on
            
        Returns:
            ProcessRewardModel: Loaded model
        """
        model = cls(path, device)
        
        # Load configuration if available
        config_path = os.path.join(path, "process_reward_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            model.verification_prompt = config.get("verification_prompt", model.verification_prompt)
        
        return model
```

### Training a Process Reward Model

To train a custom process reward model, you can use human annotations of step correctness or synthetic data:

```python
def train_process_reward_model(model_name, train_data, output_dir, epochs=3):
    """Train a process reward model.
    
    Args:
        model_name (str): Base model to fine-tune
        train_data (Dataset): Dataset with 'step' and 'label' columns
        output_dir (str): Directory to save the model
        epochs (int): Number of training epochs
    """
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Prepare training data
    verification_prompt = "Is this step correct? Yes or No: "
    
    def preprocess_function(examples):
        inputs = [verification_prompt + step for step in examples["step"]]
        targets = [" Yes" if label == 1 else " No" for label in examples["label"]]
        
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=8)
        
        return {"input_ids": model_inputs["input_ids"], 
                "attention_mask": model_inputs["attention_mask"],
                "labels": labels["input_ids"]}
    
    # Tokenize the dataset
    tokenized_dataset = train_data.map(preprocess_function, batched=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration
    config = {"verification_prompt": verification_prompt}
    with open(os.path.join(output_dir, "process_reward_config.json"), "w") as f:
        json.dump(config, f)
    
    return ProcessRewardModel.load(output_dir)
```

## Outcome Reward Models

Outcome reward models focus on evaluating the final answer or outcome of the reasoning process.

### Implementation

The `OutcomeRewardModel` class provides a foundation for outcome-level evaluation:

```python
class OutcomeRewardModel(nn.Module):
    """Model for evaluating the quality of final outcomes."""
    
    def __init__(self, model_name_or_path, device="cuda"):
        """Initialize the outcome reward model.
        
        Args:
            model_name_or_path (str): Path to the pretrained model
            device (str): Device to run the model on
        """
        super().__init__()
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Evaluation prompt template
        self.evaluation_prompt = "Is this solution correct? Yes or No: "
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Model outputs
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
    def get_reward_for_text(self, text):
        """Get reward for a single text.
        
        Args:
            text (str): The text to evaluate
            
        Returns:
            float: Reward value for the text
        """
        # Prepare input with evaluation prompt
        input_text = self.evaluation_prompt + text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.forward(inputs.input_ids, inputs.attention_mask)
            logits = outputs.logits
        
        # Get log probabilities for "Yes" and "No"
        last_token_logits = logits[0, -1]
        yes_token_id = self.tokenizer.encode(" Yes", add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode(" No", add_special_tokens=False)[0]
        
        yes_logp = torch.log_softmax(last_token_logits, dim=0)[yes_token_id].item()
        no_logp = torch.log_softmax(last_token_logits, dim=0)[no_token_id].item()
        
        # Compute reward as log(p(Yes)/p(No))
        reward = yes_logp - no_logp
        
        return reward
    
    def get_reward_for_problem_solution(self, problem, solution):
        """Get reward for a problem-solution pair.
        
        Args:
            problem (str): The problem statement
            solution (str): The proposed solution
            
        Returns:
            float: Reward value for the solution
        """
        # Combine problem and solution
        text = f"Problem: {problem}\n\nSolution: {solution}"
        return self.get_reward_for_text(text)
    
    def save(self, path):
        """Save the model and tokenizer.
        
        Args:
            path (str): Path to save the model
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save configuration
        config = {
            "evaluation_prompt": self.evaluation_prompt,
        }
        with open(os.path.join(path, "outcome_reward_config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path, device="cuda"):
        """Load the model from a path.
        
        Args:
            path (str): Path to load the model from
            device (str): Device to load the model on
            
        Returns:
            OutcomeRewardModel: Loaded model
        """
        model = cls(path, device)
        
        # Load configuration if available
        config_path = os.path.join(path, "outcome_reward_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            model.evaluation_prompt = config.get("evaluation_prompt", model.evaluation_prompt)
        
        return model
```

## Combined Reward Models

Combined reward models integrate both process and outcome evaluation for a more comprehensive assessment.

### Implementation

The `CombinedRewardModel` class provides a way to combine process and outcome evaluation:

```python
class CombinedRewardModel(nn.Module):
    """Model that combines process and outcome reward models."""
    
    def __init__(self, process_model_name=None, outcome_model_name=None, 
                 process_weight=0.7, outcome_weight=0.3, device="cuda"):
        """Initialize the combined reward model.
        
        Args:
            process_model_name (str): Path to the process reward model
            outcome_model_name (str): Path to the outcome reward model
            process_weight (float): Weight for process rewards
            outcome_weight (float): Weight for outcome rewards
            device (str): Device to run the models on
        """
        super().__init__()
        self.device = device
        self.process_weight = process_weight
        self.outcome_weight = outcome_weight
        
        # Load process reward model
        if process_model_name is not None:
            self.process_reward_model = ProcessRewardModel(process_model_name, device)
        else:
            self.process_reward_model = None
            
        # Load outcome reward model
        if outcome_model_name is not None:
            self.outcome_reward_model = OutcomeRewardModel(outcome_model_name, device)
        else:
            self.outcome_reward_model = None
            
        # Step delimiter for extracting steps
        self.step_delimiter = "\n\nStep"
        self.max_steps = None
    
    def extract_and_aggregate_steps(self, text):
        """Extract steps from text and compute rewards.
        
        Args:
            text (str): The text containing reasoning steps
            
        Returns:
            Tuple[List[str], List[float], float]: Extracted steps, step rewards, and aggregated reward
        """
        # Extract steps
        steps = extract_steps(text, self.step_delimiter, self.max_steps)
        
        # Compute rewards for each step
        if self.process_reward_model is not None:
            step_rewards = self.process_reward_model.get_rewards_for_steps(steps)
        else:
            step_rewards = [0.0] * len(steps)
            
        # Aggregate step rewards
        if len(step_rewards) > 0:
            aggregated_reward = sum(step_rewards) / len(step_rewards)
        else:
            aggregated_reward = 0.0
            
        return steps, step_rewards, aggregated_reward
    
    def get_reward_for_text(self, text):
        """Get combined reward for a single text.
        
        Args:
            text (str): The text to evaluate
            
        Returns:
            float: Combined reward value
        """
        # Get process reward
        if self.process_reward_model is not None:
            _, _, process_reward = self.extract_and_aggregate_steps(text)
        else:
            process_reward = 0.0
            
        # Get outcome reward
        if self.outcome_reward_model is not None:
            outcome_reward = self.outcome_reward_model.get_reward_for_text(text)
        else:
            outcome_reward = 0.0
            
        # Combine rewards
        combined_reward = (self.process_weight * process_reward + 
                          self.outcome_weight * outcome_reward)
        
        return combined_reward
    
    def get_rewards_for_batch(self, texts):
        """Get rewards for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to evaluate
            
        Returns:
            List[float]: List of combined rewards
        """
        return [self.get_reward_for_text(text) for text in texts]
    
    def save(self, path):
        """Save the combined reward model.
        
        Args:
            path (str): Path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save process reward model if available
        if self.process_reward_model is not None:
            process_path = os.path.join(path, "process_reward_model")
            os.makedirs(process_path, exist_ok=True)
            self.process_reward_model.save(process_path)
            
        # Save outcome reward model if available
        if self.outcome_reward_model is not None:
            outcome_path = os.path.join(path, "outcome_reward_model")
            os.makedirs(outcome_path, exist_ok=True)
            self.outcome_reward_model.save(outcome_path)
            
        # Save configuration
        config = {
            "process_weight": self.process_weight,
            "outcome_weight": self.outcome_weight,
            "step_delimiter": self.step_delimiter,
            "max_steps": self.max_steps,
        }
        with open(os.path.join(path, "combined_reward_config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path, device="cuda"):
        """Load the combined reward model from a path.
        
        Args:
            path (str): Path to load the model from
            device (str): Device to load the model on
            
        Returns:
            CombinedRewardModel: Loaded model
        """
        # Load configuration
        with open(os.path.join(path, "combined_reward_config.json"), "r") as f:
            config = json.load(f)
            
        # Create model instance
        model = cls(device=device)
        model.process_weight = config.get("process_weight", 0.7)
        model.outcome_weight = config.get("outcome_weight", 0.3)
        model.step_delimiter = config.get("step_delimiter", "\n\nStep")
        model.max_steps = config.get("max_steps", None)
        
        # Load process reward model if available
        process_path = os.path.join(path, "process_reward_model")
        if os.path.exists(process_path):
            model.process_reward_model = ProcessRewardModel.load(process_path, device)
            
        # Load outcome reward model if available
        outcome_path = os.path.join(path, "outcome_reward_model")
        if os.path.exists(outcome_path):
            model.outcome_reward_model = OutcomeRewardModel.load(outcome_path, device)
            
        return model
```

## Using Custom Reward Models with Stepwise DPO

### Integration with StepwiseDPOTrainer

To use a custom reward model with the Stepwise DPO trainer, pass it to the trainer's constructor:

```python
# Create a combined reward model
reward_model = CombinedRewardModel(
    process_model_name="path/to/process/model",
    outcome_model_name="path/to/outcome/model",
    process_weight=0.7,
    outcome_weight=0.3,
)

# Initialize trainer with reward model
trainer = StepwiseDPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    reward_model=reward_model,
)
```

### Custom Reward Calculation

The Stepwise DPO trainer will use the reward model's `get_reward_for_text` method to calculate rewards for chosen and rejected responses, and its `get_rewards_for_steps` method (if available) to calculate rewards for individual steps.

## Training Custom Reward Models

### Data Collection

To train effective reward models, you need high-quality labeled data:

1. **For Process Reward Models**: Collect examples of correct and incorrect reasoning steps
2. **For Outcome Reward Models**: Collect examples of correct and incorrect final answers

### Training Process

```python
# Prepare training data
train_data = Dataset.from_dict({
    "step": ["First, I'll solve for x by isolating the variable...", ...],
    "label": [1, 0, 1, ...],  # 1 for correct, 0 for incorrect
})

# Train process reward model
process_model = train_process_reward_model(
    model_name="gpt2",  # Or any other base model
    train_data=train_data,
    output_dir="./process_reward_model",
    epochs=3,
)

# Prepare outcome training data
outcome_data = Dataset.from_dict({
    "problem": ["Solve the equation: 3x + 7 = 22", ...],
    "solution": ["x = 5", ...],
    "label": [1, 0, 1, ...],  # 1 for correct, 0 for incorrect
})

# Train outcome reward model
# (similar to process reward model training)
```

## Best Practices

1. **Model Selection**: Choose base models that are appropriate for the task domain (e.g., math-specialized models for mathematical reasoning)

2. **Data Quality**: Use high-quality labeled data for training reward models, preferably with expert annotations

3. **Prompt Engineering**: Carefully design verification prompts to elicit accurate evaluations

4. **Balancing Weights**: Experiment with different weights for process and outcome rewards to find the optimal balance

5. **Evaluation**: Regularly evaluate reward models on held-out data to ensure they provide meaningful signals

## Advanced Techniques

### Reward Shaping

You can implement more sophisticated reward shaping techniques in your custom reward models:

```python
def shaped_reward(base_reward, confidence, complexity):
    """Shape the reward based on additional factors.
    
    Args:
        base_reward (float): Base reward value
        confidence (float): Model's confidence in the evaluation
        complexity (float): Complexity of the reasoning step
        
    Returns:
        float: Shaped reward
    """
    # Boost rewards for complex steps that are correct
    if base_reward > 0:
        return base_reward * (1 + 0.5 * complexity)
    else:
        return base_reward
```

### Ensemble Reward Models

You can create ensemble reward models that combine evaluations from multiple models:

```python
class EnsembleRewardModel(nn.Module):
    """Ensemble of multiple reward models."""
    
    def __init__(self, model_paths, weights=None):
        """Initialize the ensemble reward model.
        
        Args:
            model_paths (List[str]): Paths to individual reward models
            weights (List[float], optional): Weights for each model
        """
        super().__init__()
        self.models = [CombinedRewardModel.load(path) for path in model_paths]
        self.weights = weights or [1.0 / len(model_paths)] * len(model_paths)
    
    def get_reward_for_text(self, text):
        """Get ensemble reward for a single text.
        
        Args:
            text (str): The text to evaluate
            
        Returns:
            float: Ensemble reward value
        """
        rewards = [model.get_reward_for_text(text) for model in self.models]
        ensemble_reward = sum(w * r for w, r in zip(self.weights, rewards))
        return ensemble_reward
```

## References

1. "Let's Verify Step by Step" (arXiv:2408.15240v1)
2. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)
3. Process Supervision: Going Beyond Step-by-Step Instructions (Lightman et al., 2023)
4. Training language models with language feedback (Scheurer et al., 2023)