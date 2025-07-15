import jax
import optax
import jax.numpy as jnp
import haiku as hk

from rl_zoo.utils.model import build_mlp
from collections import namedtuple
from typing import List, Tuple, NamedTuple

WorldModelParameters = namedtuple("WorldModelParameters", "dynamics reward")
WorldModelOutput = namedtuple("WorldModelOutput", "next_obs reward done")
Loss = namedtuple("Loss", "dynamics_loss reward_loss")


class WorldModel:
    """
    A world model that learns environment dynamics and rewards.
    Used for model-based RL algorithms like MBPO.
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hiddens: List[int] = [200, 200, 200, 200],
                 learning_rate: float = 1e-3,
                 ensemble_size: int = 7,
                 elite_size: int = 5):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.elite_size = elite_size
        
        # Build ensemble of dynamics models
        self.dynamics_models = []
        self.reward_models = []
        
        for _ in range(ensemble_size):
            # Dynamics model predicts next observation and done flag
            dynamics_model = build_mlp(hiddens, obs_dim + 1)  # +1 for done flag
            self.dynamics_models.append(dynamics_model)
            
            # Reward model predicts immediate reward
            reward_model = build_mlp(hiddens, 1)
            self.reward_models.append(reward_model)
        
        self.optimizer = optax.adam(learning_rate)
        
        # JIT compile functions
        self.update_parameters = jax.jit(self.update_parameters)
        self.predict = jax.jit(self.predict)
        self.predict_ensemble = jax.jit(self.predict_ensemble)

    def initial_parameters(self, rng) -> WorldModelParameters:
        """Initialize parameters for all models in the ensemble"""
        sample_input = jnp.zeros((1, self.obs_dim + self.action_dim))
        
        dynamics_params = []
        reward_params = []
        
        for i in range(self.ensemble_size):
            rng, subkey = jax.random.split(rng)
            
            # Initialize dynamics model
            dynamics_param = self.dynamics_models[i].init(subkey, sample_input)
            dynamics_params.append(dynamics_param)
            
            # Initialize reward model
            rng, subkey = jax.random.split(rng)
            reward_param = self.reward_models[i].init(subkey, sample_input)
            reward_params.append(reward_param)
        
        return WorldModelParameters(dynamics_params, reward_params)

    def initial_optimizer(self, params: WorldModelParameters):
        """Initialize optimizer states for all models"""
        dynamics_opt_states = []
        reward_opt_states = []
        
        for i in range(self.ensemble_size):
            dynamics_opt_state = self.optimizer.init(params.dynamics[i])
            dynamics_opt_states.append(dynamics_opt_state)
            
            reward_opt_state = self.optimizer.init(params.reward[i])
            reward_opt_states.append(reward_opt_state)
        
        return WorldModelParameters(dynamics_opt_states, reward_opt_states)

    def dynamics_loss(self, dynamics_params, reward_params, obs, action, next_obs, reward, done):
        """Compute loss for a single model in the ensemble"""
        input_data = jnp.concatenate([obs, action], axis=1)
        
        # Predict next observation and done flag
        dynamics_pred = self.dynamics_models[0].apply(dynamics_params, input_data)
        next_obs_pred = dynamics_pred[:, :self.obs_dim]
        done_pred = dynamics_pred[:, self.obs_dim]
        
        # Predict reward
        reward_pred = self.reward_models[0].apply(reward_params, input_data)
        
        # Compute losses
        dynamics_loss = jnp.mean(jnp.square(next_obs_pred - next_obs))
        reward_loss = jnp.mean(jnp.square(reward_pred - reward))
        done_loss = jnp.mean(jnp.square(done_pred - done))
        
        total_loss = dynamics_loss + reward_loss + done_loss
        
        return total_loss, (dynamics_loss, reward_loss)

    def update_parameters(self,
                         params: WorldModelParameters,
                         opt_states: WorldModelParameters,
                         obs: jnp.ndarray,
                         action: jnp.ndarray,
                         next_obs: jnp.ndarray,
                         reward: jnp.ndarray,
                         done: jnp.ndarray) -> Tuple[WorldModelParameters, WorldModelParameters, Loss]:
        """Update all models in the ensemble"""
        
        new_dynamics_params = []
        new_reward_params = []
        new_dynamics_opt_states = []
        new_reward_opt_states = []
        
        total_dynamics_loss = 0.0
        total_reward_loss = 0.0
        
        for i in range(self.ensemble_size):
            # Create loss function for this specific model
            def model_loss(dynamics_params, reward_params):
                input_data = jnp.concatenate([obs, action], axis=1)
                
                # Predict next observation and done flag
                dynamics_pred = self.dynamics_models[i].apply(dynamics_params, input_data)
                next_obs_pred = dynamics_pred[:, :self.obs_dim]
                done_pred = dynamics_pred[:, self.obs_dim]
                
                # Predict reward
                reward_pred = self.reward_models[i].apply(reward_params, input_data)
                
                # Compute losses
                dynamics_loss = jnp.mean(jnp.square(next_obs_pred - next_obs))
                reward_loss = jnp.mean(jnp.square(reward_pred - reward))
                done_loss = jnp.mean(jnp.square(done_pred - done))
                
                total_loss = dynamics_loss + reward_loss + done_loss
                
                return total_loss, (dynamics_loss, reward_loss)
            
            # Compute gradients
            grad_fn = jax.value_and_grad(model_loss, argnums=(0, 1), has_aux=True)
            (loss, (dynamics_loss, reward_loss)), (dynamics_grads, reward_grads) = grad_fn(
                params.dynamics[i], params.reward[i]
            )
            
            # Update dynamics model
            dynamics_updates, dynamics_opt_state = self.optimizer.update(
                dynamics_grads, opt_states.dynamics[i]
            )
            new_dynamics_param = optax.apply_updates(params.dynamics[i], dynamics_updates)
            
            # Update reward model
            reward_updates, reward_opt_state = self.optimizer.update(
                reward_grads, opt_states.reward[i]
            )
            new_reward_param = optax.apply_updates(params.reward[i], reward_updates)
            
            new_dynamics_params.append(new_dynamics_param)
            new_reward_params.append(new_reward_param)
            new_dynamics_opt_states.append(dynamics_opt_state)
            new_reward_opt_states.append(reward_opt_state)
            
            total_dynamics_loss += dynamics_loss
            total_reward_loss += reward_loss
        
        # Average losses across ensemble
        avg_dynamics_loss = total_dynamics_loss / self.ensemble_size
        avg_reward_loss = total_reward_loss / self.ensemble_size
        
        return (
            WorldModelParameters(new_dynamics_params, new_reward_params),
            WorldModelParameters(new_dynamics_opt_states, new_reward_opt_states),
            Loss(avg_dynamics_loss, avg_reward_loss)
        )

    def predict(self, params: WorldModelParameters, obs: jnp.ndarray, action: jnp.ndarray, rng) -> WorldModelOutput:
        """Predict next state using a random model from the ensemble"""
        # Select random model from ensemble
        model_idx = jax.random.randint(rng, (), 0, self.ensemble_size)
        
        input_data = jnp.concatenate([obs, action], axis=1)
        
        # Predict dynamics
        dynamics_pred = self.dynamics_models[model_idx].apply(params.dynamics[model_idx], input_data)
        next_obs = dynamics_pred[:, :self.obs_dim]
        done = dynamics_pred[:, self.obs_dim]
        
        # Predict reward
        reward = self.reward_models[model_idx].apply(params.reward[model_idx], input_data)
        
        return WorldModelOutput(next_obs, reward, done)

    def predict_ensemble(self, params: WorldModelParameters, obs: jnp.ndarray, action: jnp.ndarray) -> WorldModelOutput:
        """Predict next state using all models in the ensemble and return statistics"""
        input_data = jnp.concatenate([obs, action], axis=1)
        
        next_obs_preds = []
        reward_preds = []
        done_preds = []
        
        for i in range(self.ensemble_size):
            # Predict dynamics
            dynamics_pred = self.dynamics_models[i].apply(params.dynamics[i], input_data)
            next_obs_pred = dynamics_pred[:, :self.obs_dim]
            done_pred = dynamics_pred[:, self.obs_dim]
            
            # Predict reward
            reward_pred = self.reward_models[i].apply(params.reward[i], input_data)
            
            next_obs_preds.append(next_obs_pred)
            reward_preds.append(reward_pred)
            done_preds.append(done_pred)
        
        # Stack predictions
        next_obs_ensemble = jnp.stack(next_obs_preds, axis=0)
        reward_ensemble = jnp.stack(reward_preds, axis=0)
        done_ensemble = jnp.stack(done_preds, axis=0)
        
        # Return mean predictions
        return WorldModelOutput(
            jnp.mean(next_obs_ensemble, axis=0),
            jnp.mean(reward_ensemble, axis=0),
            jnp.mean(done_ensemble, axis=0)
        )

    def rollout(self, params: WorldModelParameters, obs: jnp.ndarray, policy_fn, horizon: int, rng) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Rollout using the world model"""
        batch_size = obs.shape[0]
        
        # Initialize storage
        observations = [obs]
        actions = []
        rewards = []
        
        current_obs = obs
        
        for step in range(horizon):
            rng, subkey = jax.random.split(rng)
            
            # Get action from policy
            action = policy_fn(current_obs, subkey)
            actions.append(action)
            
            # Predict next state
            rng, subkey = jax.random.split(rng)
            prediction = self.predict(params, current_obs, action, subkey)
            
            current_obs = prediction.next_obs
            observations.append(current_obs)
            rewards.append(prediction.reward)
        
        return (
            jnp.stack(observations[:-1], axis=0),  # Remove last obs
            jnp.stack(actions, axis=0),
            jnp.stack(rewards, axis=0)
        )