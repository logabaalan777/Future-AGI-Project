# Reward Aggregation Strategies in Stepwise DPO

This document explains the different reward aggregation strategies implemented in the Stepwise DPO trainer, based on the "Let's Verify Step by Step" methodology (arXiv:2408.15240v1).

## Overview

Stepwise DPO optimizes individual reasoning steps rather than just the final answer. This requires aggregating step-level rewards into a single reward signal for training. The choice of aggregation strategy can significantly impact the model's learning behavior and the quality of its reasoning.

## Available Strategies

### 1. Weighted Sum

**Formula**: $R_{agg} = \sum_{i=1}^{n} w_i \cdot R_i$

The weighted sum strategy computes a linear combination of step rewards, where each step can be assigned a different weight. This is the most flexible strategy and allows for emphasizing certain steps over others.

**Parameters**:
- `step_weights`: Optional list of weights for each step. If not provided, uniform weights are used.

**Use cases**:
- When certain steps are known to be more important than others
- When you want to emphasize early or late steps in the reasoning process
- As a general-purpose strategy when the relative importance of steps is understood

**Example**:
```python
# With uniform weights (default)
aggregated_reward = aggregate_rewards([0.8, 0.6, 0.9, 0.7], "weighted_sum")

# With custom weights (must sum to 1)
aggregated_reward = aggregate_rewards([0.8, 0.6, 0.9, 0.7], "weighted_sum", [0.4, 0.2, 0.3, 0.1])
```

### 2. Min Step

**Formula**: $R_{agg} = \min(R_1, R_2, ..., R_n)$

The min step strategy takes the minimum reward across all steps. This enforces a strong penalty for any incorrect step in the reasoning process.

**Use cases**:
- When a single incorrect step can invalidate the entire reasoning process
- For tasks where correctness at every step is critical
- When you want to encourage the model to be consistently correct

**Example**:
```python
aggregated_reward = aggregate_rewards([0.8, 0.6, 0.9, 0.7], "min_step")
# Result: 0.6 (the minimum value)
```

### 3. Harmonic Mean

**Formula**: $R_{agg} = \frac{n}{\sum_{i=1}^{n} \frac{1}{R_i + \epsilon}}$

The harmonic mean gives more weight to lower values, making it sensitive to poor-performing steps while still considering all steps. A small epsilon is added to prevent division by zero.

**Use cases**:
- When you want to penalize poor steps more than reward good steps
- For tasks requiring consistent reasoning quality
- When a balance between min and average is desired

**Example**:
```python
aggregated_reward = aggregate_rewards([0.8, 0.6, 0.9, 0.7], "harmonic_mean")
```

### 4. Geometric Mean

**Formula**: $R_{agg} = \left(\prod_{i=1}^{n} (R_i + \epsilon)\right)^{1/n} - \epsilon$

The geometric mean is less sensitive to outliers than the arithmetic mean but more balanced than the min strategy. A small epsilon is added to handle zero or negative rewards.

**Use cases**:
- When you want a balanced approach that still penalizes poor steps
- For tasks where the quality of reasoning should be consistent
- When extreme values should be dampened but still considered

**Example**:
```python
aggregated_reward = aggregate_rewards([0.8, 0.6, 0.9, 0.7], "geometric_mean")
```

### 5. Custom

The Stepwise DPO trainer also supports custom aggregation strategies by providing a callable function.

**Example**:
```python
def my_custom_aggregation(rewards, **kwargs):
    # Implement custom logic here
    return some_aggregated_value

aggregated_reward = aggregate_rewards([0.8, 0.6, 0.9, 0.7], my_custom_aggregation)
```

## Choosing the Right Strategy

The choice of aggregation strategy depends on the specific requirements of your task:

1. **Weighted Sum**: Use when you have prior knowledge about the relative importance of different steps or want to emphasize certain parts of the reasoning process.

2. **Min Step**: Use when a single incorrect step can invalidate the entire reasoning process, such as in mathematical proofs or logical deductions.

3. **Harmonic Mean**: Use when you want to penalize poor steps more heavily while still considering all steps, suitable for tasks requiring consistent reasoning quality.

4. **Geometric Mean**: Use when you want a balanced approach that dampens the effect of outliers but still considers all steps.

5. **Custom**: Implement when you have specific requirements not covered by the standard strategies.

## Implementation Details

The aggregation strategies are implemented in the `aggregate_rewards` function in `trainer/utils.py`. The function takes a list of rewards, a strategy name, and optional parameters specific to the strategy.

```python
def aggregate_rewards(rewards, strategy, weights=None, epsilon=1e-10):
    """Aggregate step-level rewards using the specified strategy.
    
    Args:
        rewards (List[float]): List of rewards for each step
        strategy (str or callable): Strategy to use for aggregation
        weights (List[float], optional): Weights for weighted sum strategy
        epsilon (float, optional): Small value to prevent division by zero
        
    Returns:
        float: Aggregated reward
    """
    # Implementation details...
```

## Experimental Results

Different aggregation strategies can lead to different training dynamics and model behaviors. Here are some general observations:

- **Weighted Sum** with uniform weights tends to produce models that are good on average but might occasionally make significant errors.

- **Min Step** produces models that are more conservative and make fewer significant errors, but might be less creative or flexible.

- **Harmonic Mean** and **Geometric Mean** tend to balance between the extremes, producing models that are generally reliable while still maintaining some flexibility.

- Custom weighting in **Weighted Sum** can be particularly effective when tailored to specific task requirements, such as giving more weight to critical steps in the reasoning process.

## References

1. "Let's Verify Step by Step" (arXiv:2408.15240v1)
2. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)