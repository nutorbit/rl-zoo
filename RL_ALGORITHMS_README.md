# RL Zoo - Reinforcement Learning Algorithms

This repository contains implementations of state-of-the-art reinforcement learning algorithms using JAX, including SAC, DDPG, and Model-Based RL (MBPO).

## Implemented Algorithms

### 1. SAC (Soft Actor-Critic)
- **File**: `rl_zoo/off_policy/sac.py`
- **Type**: Off-policy, Actor-Critic
- **Key Features**:
  - Automatic entropy tuning
  - Twin critic networks (Q1, Q2)
  - Reparameterization trick for continuous actions
  - Soft target updates
  - Maximum entropy objective

### 2. DDPG (Deep Deterministic Policy Gradient)
- **File**: `rl_zoo/off_policy/ddpg.py` (already implemented)
- **Type**: Off-policy, Actor-Critic
- **Key Features**:
  - Deterministic policy gradient
  - Target networks for stability
  - Experience replay
  - Continuous action spaces

### 3. MBPO (Model-Based Policy Optimization)
- **Files**: 
  - `rl_zoo/model_based/world_model.py` (Environment dynamics model)
  - `rl_zoo/model_based/mbpo.py` (Complete MBPO algorithm)
- **Type**: Model-based RL
- **Key Features**:
  - Ensemble of world models (default: 7 models)
  - Combines model-based and model-free learning
  - Uses SAC as the model-free component
  - Short horizon rollouts (default: 1 step)
  - Dyna-style synthetic experience generation

## Architecture

All algorithms follow a consistent JAX-based architecture:

```python
# Common pattern for all algorithms
class Algorithm:
    def __init__(self, obs_dim, action_dim, ...):
        # Initialize networks and optimizers
        
    def initial_parameters(self, rng):
        # Initialize all network parameters
        
    def initial_optimizer(self, params):
        # Initialize optimizer states
        
    def update_parameters(self, params, opt_states, data, rng):
        # Update algorithm parameters
        
    def get_action(self, params, obs, rng):
        # Get action from policy
```

## Usage Examples

### Quick Start

```python
import jax
from rl_zoo import SAC, DDPG, MBPO, ReplayBuffer

# Initialize algorithm
rng = jax.random.PRNGKey(42)
agent = SAC(obs_dim=3, action_dim=1)

# Initialize parameters and optimizer
params = agent.initial_parameters(rng)
opt_states = agent.initial_optimizer(params)

# Training loop
replay_buffer = ReplayBuffer(size=100000)
# ... collect experience and update agent
```

### SAC Example

```python
import jax
import jax.numpy as jnp
from rl_zoo import SAC, ReplayBuffer
from rl_zoo.utils.common import Transition

# Initialize SAC
rng = jax.random.PRNGKey(42)
agent = SAC(obs_dim=3, action_dim=1)
params = agent.initial_parameters(rng)
opt_states = agent.initial_optimizer(params)

# Training step
data = replay_buffer.sample_experience(256)
transition = Transition(*data)
rng, subkey = jax.random.split(rng)
params, opt_states, loss = agent.update_parameters(params, opt_states, transition, subkey)

# Get action
rng, subkey = jax.random.split(rng)
action = agent.get_action(params, obs, subkey).action
```

### MBPO Example

```python
import jax
from rl_zoo import MBPO, ReplayBuffer

# Initialize MBPO
rng = jax.random.PRNGKey(42)
agent = MBPO(obs_dim=3, action_dim=1, rollout_horizon=1)
params = agent.initial_parameters(rng)
opt_states = agent.initial_optimizer(params)

# Training step (updates both world model and SAC)
rng, subkey = jax.random.split(rng)
params, opt_states, losses = agent.train_step(params, opt_states, replay_buffer, 256, subkey)
```

## Key Features

### JAX-First Design
- All algorithms use JAX for automatic differentiation and JIT compilation
- Functional programming paradigm with immutable parameters
- GPU/TPU acceleration ready

### Modular Architecture
- Clean separation between algorithms, models, and utilities
- Easy to extend and modify
- Consistent API across all algorithms

### Production Ready
- Comprehensive parameter management
- Proper optimizer state handling
- Memory-efficient replay buffers
- Extensive documentation and examples

## Dependencies

```toml
jax = "^0.3.23"
jaxlib = "^0.3.22"
dm-haiku = "^0.0.8"
optax = "^0.1.3"
gym = {extras = ["all"], version = "^0.26.2"}
```

## File Structure

```
rl_zoo/
├── __init__.py                 # Main imports
├── off_policy/
│   ├── __init__.py            # Off-policy algorithms
│   ├── sac.py                 # SAC implementation
│   ├── ddpg.py                # DDPG implementation
│   ├── dqn.py                 # DQN implementation
│   └── replay.py              # Replay buffer
├── model_based/
│   ├── __init__.py            # Model-based algorithms
│   ├── world_model.py         # Environment dynamics model
│   └── mbpo.py                # MBPO implementation
└── utils/
    ├── common.py              # Common utilities
    ├── model.py               # Neural network utilities
    └── roller.py              # Rollout utilities
```

## Algorithm Comparison

| Algorithm | Type | Sample Efficiency | Stability | Continuous Actions | Discrete Actions |
|-----------|------|-------------------|-----------|-------------------|------------------|
| SAC       | Model-free | High | High | ✅ | ❌ |
| DDPG      | Model-free | Medium | Medium | ✅ | ❌ |
| MBPO      | Model-based | Very High | High | ✅ | ❌ |

## Advanced Usage

### Custom Network Architectures

```python
# Custom hidden layers
agent = SAC(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hiddens=[512, 512, 256],  # Custom architecture
    learning_rate=1e-4
)
```

### Hyperparameter Tuning

```python
# SAC with custom hyperparameters
agent = SAC(
    obs_dim=obs_dim,
    action_dim=action_dim,
    gamma=0.99,           # Discount factor
    tau=0.005,            # Soft update coefficient
    learning_rate=3e-4,   # Learning rate
    auto_alpha=True,      # Automatic entropy tuning
    target_entropy=-action_dim  # Target entropy
)

# MBPO with custom model settings
agent = MBPO(
    obs_dim=obs_dim,
    action_dim=action_dim,
    ensemble_size=7,      # Number of models in ensemble
    rollout_horizon=1,    # Rollout length
    real_ratio=0.5,       # Ratio of real vs synthetic data
    rollout_batch_size=400  # Batch size for rollouts
)
```

## Testing

Run the example training script:

```bash
python examples/train_example.py
```

This will train all three algorithms on the Pendulum environment and display their performance.

## References

1. **SAC**: Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." ICML 2018.
2. **DDPG**: Lillicrap, T., et al. "Continuous control with deep reinforcement learning." ICLR 2016.
3. **MBPO**: Janner, M., et al. "When to trust your model: Model-based policy optimization." NeurIPS 2019.

## Contributing

Feel free to extend these implementations with additional algorithms or improvements. The modular design makes it easy to add new algorithms while maintaining consistency.

## License

This project follows the same license as the parent repository.