import jax
import jax.numpy as jnp
import gym
import numpy as np
from tqdm import tqdm

from rl_zoo import SAC, DDPG, MBPO, ReplayBuffer


def train_sac_example():
    """Example of training SAC on a continuous control environment"""
    print("Training SAC...")
    
    # Create environment
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    rng = jax.random.PRNGKey(42)
    agent = SAC(obs_dim=obs_dim, action_dim=action_dim)
    
    # Initialize parameters and optimizer
    params = agent.initial_parameters(rng)
    opt_states = agent.initial_optimizer(params)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(size=100000)
    
    # Training loop
    total_steps = 0
    episode_rewards = []
    
    for episode in tqdm(range(100)):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        episode_reward = 0
        
        for step in range(200):
            # Get action
            rng, subkey = jax.random.split(rng)
            if total_steps < 1000:  # Random exploration
                action = env.action_space.sample()
            else:
                action = agent.get_action(params, obs, subkey).action[0]
            
            # Take step
            next_obs, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Store experience
            replay_buffer.add_experience(obs, action, reward, next_obs, done)
            
            # Update agent
            if replay_buffer.current_size > 1000:
                data = replay_buffer.sample_experience(256)
                from rl_zoo.utils.common import Transition
                transition = Transition(*data)
                
                rng, subkey = jax.random.split(rng)
                params, opt_states, loss = agent.update_parameters(
                    params, opt_states, transition, subkey
                )
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    
    print(f"SAC training complete. Final 10 episodes average: {np.mean(episode_rewards[-10:]):.2f}")
    return params, agent


def train_ddpg_example():
    """Example of training DDPG on a continuous control environment"""
    print("Training DDPG...")
    
    # Create environment
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    rng = jax.random.PRNGKey(42)
    agent = DDPG(obs_dim=obs_dim, action_dim=action_dim)
    
    # Initialize parameters and optimizer
    params = agent.initial_parameters(rng)
    opt_states = agent.initial_optimizer(params)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(size=100000)
    
    # Training loop
    total_steps = 0
    episode_rewards = []
    
    for episode in tqdm(range(100)):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        episode_reward = 0
        
        for step in range(200):
            # Get action
            rng, subkey = jax.random.split(rng)
            if total_steps < 1000:  # Random exploration
                action = env.action_space.sample()
            else:
                action = agent.get_random_action(params, obs, subkey).action[0]
            
            # Take step
            next_obs, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Store experience
            replay_buffer.add_experience(obs, action, reward, next_obs, done)
            
            # Update agent
            if replay_buffer.current_size > 1000:
                data = replay_buffer.sample_experience(256)
                from rl_zoo.utils.common import Transition
                transition = Transition(*data)
                
                params, opt_states, loss = agent.update_parameters(
                    params, opt_states, transition
                )
                
                # Update target networks periodically
                if total_steps % 100 == 0:
                    params = agent.update_target(params)
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    
    print(f"DDPG training complete. Final 10 episodes average: {np.mean(episode_rewards[-10:]):.2f}")
    return params, agent


def train_mbpo_example():
    """Example of training MBPO on a continuous control environment"""
    print("Training MBPO...")
    
    # Create environment
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    rng = jax.random.PRNGKey(42)
    agent = MBPO(obs_dim=obs_dim, action_dim=action_dim, rollout_horizon=1)
    
    # Initialize parameters and optimizer
    params = agent.initial_parameters(rng)
    opt_states = agent.initial_optimizer(params)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(size=100000)
    
    # Training loop
    total_steps = 0
    episode_rewards = []
    
    for episode in tqdm(range(100)):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        episode_reward = 0
        
        for step in range(200):
            # Get action
            rng, subkey = jax.random.split(rng)
            if total_steps < 1000:  # Random exploration
                action = env.action_space.sample()
            else:
                action = agent.get_action(params, obs, subkey).action[0]
            
            # Take step
            next_obs, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Store experience
            replay_buffer.add_experience(obs, action, reward, next_obs, done)
            
            # Update agent with MBPO
            if replay_buffer.current_size > 1000:
                rng, subkey = jax.random.split(rng)
                params, opt_states, losses = agent.train_step(
                    params, opt_states, replay_buffer, 256, subkey
                )
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    
    print(f"MBPO training complete. Final 10 episodes average: {np.mean(episode_rewards[-10:]):.2f}")
    return params, agent


if __name__ == "__main__":
    print("RL Zoo - Algorithm Examples")
    print("=" * 50)
    
    # Train SAC
    sac_params, sac_agent = train_sac_example()
    print()
    
    # Train DDPG  
    ddpg_params, ddpg_agent = train_ddpg_example()
    print()
    
    # Train MBPO
    mbpo_params, mbpo_agent = train_mbpo_example()
    print()
    
    print("All algorithms trained successfully!")
    print("You can now use these trained agents for inference.")