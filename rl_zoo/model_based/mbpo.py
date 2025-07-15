import jax
import jax.numpy as jnp
import numpy as np

from rl_zoo.off_policy.sac import SAC
from rl_zoo.model_based.world_model import WorldModel
from rl_zoo.off_policy.replay import ReplayBuffer
from rl_zoo.utils.common import Transition
from collections import namedtuple
from typing import List, Tuple, NamedTuple, Callable

MBPOParameters = namedtuple("MBPOParameters", "sac world_model")
MBPOOptimizerState = namedtuple("MBPOOptimizerState", "sac world_model")
MBPOLoss = namedtuple("MBPOLoss", "sac_loss world_model_loss")


class MBPO:
    """
    Implementation of Model-Based Policy Optimization (MBPO)
    Reference:
        https://arxiv.org/abs/1906.08253
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 sac_hiddens: List[int] = [256, 256],
                 model_hiddens: List[int] = [200, 200, 200, 200],
                 gamma: float = 0.99,
                 sac_learning_rate: float = 3e-4,
                 model_learning_rate: float = 1e-3,
                 tau: float = 0.005,
                 target_entropy: float = None,
                 auto_alpha: bool = True,
                 ensemble_size: int = 7,
                 elite_size: int = 5,
                 rollout_horizon: int = 1,
                 rollout_batch_size: int = 400,
                 real_ratio: float = 0.5,
                 model_retain_epochs: int = 1):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rollout_horizon = rollout_horizon
        self.rollout_batch_size = rollout_batch_size
        self.real_ratio = real_ratio
        self.model_retain_epochs = model_retain_epochs
        
        # Initialize SAC agent
        self.sac = SAC(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hiddens=sac_hiddens,
            gamma=gamma,
            learning_rate=sac_learning_rate,
            tau=tau,
            target_entropy=target_entropy,
            auto_alpha=auto_alpha
        )
        
        # Initialize world model
        self.world_model = WorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hiddens=model_hiddens,
            learning_rate=model_learning_rate,
            ensemble_size=ensemble_size,
            elite_size=elite_size
        )
        
        # Model replay buffer for synthetic data
        self.model_buffer = ReplayBuffer(size=100000)
        
        # JIT compile functions
        self.update_sac = jax.jit(self.update_sac)
        self.update_world_model = jax.jit(self.update_world_model)

    def initial_parameters(self, rng) -> MBPOParameters:
        """Initialize parameters for both SAC and world model"""
        rng1, rng2 = jax.random.split(rng, 2)
        
        sac_params = self.sac.initial_parameters(rng1)
        world_model_params = self.world_model.initial_parameters(rng2)
        
        return MBPOParameters(sac_params, world_model_params)

    def initial_optimizer(self, params: MBPOParameters) -> MBPOOptimizerState:
        """Initialize optimizer states"""
        sac_opt_state = self.sac.initial_optimizer(params.sac)
        world_model_opt_state = self.world_model.initial_optimizer(params.world_model)
        
        return MBPOOptimizerState(sac_opt_state, world_model_opt_state)

    def update_world_model(self,
                          params: MBPOParameters,
                          opt_states: MBPOOptimizerState,
                          data: Transition) -> Tuple[MBPOParameters, MBPOOptimizerState, any]:
        """Update world model parameters"""
        obs, action, reward, next_obs, done = data
        
        new_world_model_params, new_world_model_opt_state, world_model_loss = self.world_model.update_parameters(
            params.world_model, opt_states.world_model, obs, action, next_obs, reward, done
        )
        
        return (
            MBPOParameters(params.sac, new_world_model_params),
            MBPOOptimizerState(opt_states.sac, new_world_model_opt_state),
            world_model_loss
        )

    def update_sac(self,
                   params: MBPOParameters,
                   opt_states: MBPOOptimizerState,
                   data: Transition,
                   rng) -> Tuple[MBPOParameters, MBPOOptimizerState, any]:
        """Update SAC parameters"""
        new_sac_params, new_sac_opt_state, sac_loss = self.sac.update_parameters(
            params.sac, opt_states.sac, data, rng
        )
        
        return (
            MBPOParameters(new_sac_params, params.world_model),
            MBPOOptimizerState(new_sac_opt_state, opt_states.world_model),
            sac_loss
        )

    def generate_model_data(self,
                           params: MBPOParameters,
                           real_buffer: ReplayBuffer,
                           rng) -> None:
        """Generate synthetic data using the world model"""
        # Sample initial states from real buffer
        if real_buffer.current_size < self.rollout_batch_size:
            return
            
        # Sample starting states
        real_obs, _, _, _, _ = real_buffer.sample_experience(self.rollout_batch_size)
        
        # Define policy function for rollouts
        def policy_fn(obs, rng_key):
            return self.sac.get_action(params.sac, obs, rng_key).action
        
        # Generate rollouts
        rng, subkey = jax.random.split(rng)
        obs_seq, action_seq, reward_seq = self.world_model.rollout(
            params.world_model, real_obs, policy_fn, self.rollout_horizon, subkey
        )
        
        # Add synthetic data to model buffer
        self.model_buffer.reset()  # Clear previous model data
        
        for step in range(self.rollout_horizon):
            for i in range(self.rollout_batch_size):
                obs = obs_seq[step, i]
                action = action_seq[step, i]
                reward = reward_seq[step, i]
                
                if step < self.rollout_horizon - 1:
                    next_obs = obs_seq[step + 1, i]
                    done = 0.0  # Model rollouts don't terminate early
                else:
                    # Use world model to predict next state for last step
                    rng, subkey = jax.random.split(rng)
                    prediction = self.world_model.predict(params.world_model, obs, action, subkey)
                    next_obs = prediction.next_obs
                    done = prediction.done
                
                self.model_buffer.add_experience(
                    np.array(obs), np.array(action), float(reward), 
                    np.array(next_obs), float(done)
                )

    def train_step(self,
                   params: MBPOParameters,
                   opt_states: MBPOOptimizerState,
                   real_buffer: ReplayBuffer,
                   batch_size: int,
                   rng) -> Tuple[MBPOParameters, MBPOOptimizerState, MBPOLoss]:
        """Complete MBPO training step"""
        
        # 1. Update world model with real data
        if real_buffer.current_size >= batch_size:
            real_data = real_buffer.sample_experience(batch_size)
            real_transition = Transition(*real_data)
            
            params, opt_states, world_model_loss = self.update_world_model(
                params, opt_states, real_transition
            )
        else:
            world_model_loss = None
        
        # 2. Generate synthetic data
        rng, subkey = jax.random.split(rng)
        self.generate_model_data(params, real_buffer, subkey)
        
        # 3. Update SAC with mixture of real and synthetic data
        sac_loss = None
        if real_buffer.current_size >= batch_size and self.model_buffer.current_size >= batch_size:
            # Sample from both buffers
            real_batch_size = int(batch_size * self.real_ratio)
            model_batch_size = batch_size - real_batch_size
            
            real_data = real_buffer.sample_experience(real_batch_size)
            model_data = self.model_buffer.sample_experience(model_batch_size)
            
            # Combine real and synthetic data
            combined_data = tuple(
                jnp.concatenate([real_batch, model_batch], axis=0)
                for real_batch, model_batch in zip(real_data, model_data)
            )
            
            combined_transition = Transition(*combined_data)
            
            # Update SAC
            rng, subkey = jax.random.split(rng)
            params, opt_states, sac_loss = self.update_sac(
                params, opt_states, combined_transition, subkey
            )
        
        return params, opt_states, MBPOLoss(sac_loss, world_model_loss)

    def get_action(self, params: MBPOParameters, obs: jnp.ndarray, rng):
        """Get action from SAC policy"""
        return self.sac.get_action(params.sac, obs, rng)

    def get_deterministic_action(self, params: MBPOParameters, obs: jnp.ndarray):
        """Get deterministic action for evaluation"""
        return self.sac.get_deterministic_action(params.sac, obs)

    def get_random_action(self, params: MBPOParameters, obs: jnp.ndarray, rng):
        """Get random action for exploration"""
        return self.sac.get_random_action(params.sac, obs, rng)