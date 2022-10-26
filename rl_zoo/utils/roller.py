import jax
import numpy as np

from tqdm import tqdm

from rl_zoo.off_policy.dqn import DQN
from rl_zoo.off_policy.replay import ReplayBuffer


class OffPolicyRoller:
    def __init__(self,
                 policy,
                 env,
                 warm_start: int = 1000,
                 max_timesteps: int = 100_000,
                 max_timesteps_per_episode: int = 200,
                 epsilon: float = 0.1,
                 replay_size: int = 10_000,
                 update_every: int = 5,
                 batch_size: int = 32,
                 ):

        self.policy = policy
        self.env = env
        self.warm_start = warm_start
        self.max_timesteps = max_timesteps
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.epsilon = epsilon
        self.replay_size = replay_size
        self.update_every = update_every
        self.batch_size = batch_size

    def run(self):
        rb = ReplayBuffer(self.replay_size)

        rng = jax.random.PRNGKey(1)
        rng, rng_init = jax.random.split(rng)
        params = self.policy.initial_parameters(rng_init)
        opt_state = self.policy.initial_optimizer(params.q)

        obs, _ = self.env.reset()

        timesteps = 0
        ep_reward = 0
        ep_timesteps = 0

        with tqdm(total=self.max_timesteps) as pbar:
            while timesteps < self.max_timesteps:
                # select action
                if timesteps <= self.warm_start or np.random.uniform() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    rng, rng_action = jax.random.split(rng)
                    action = self.policy.get_action(params, obs, rng_action)

                    if not isinstance(action.action, np.ndarray):
                        action = np.array(action.action)[0]

                # step
                next_obs, reward, done, _, _ = self.env.step(action)
                rb.add_experience(obs, action, reward, next_obs, done)

                if timesteps % self.update_every == 0:
                    data = rb.sample_experience(self.batch_size)
                    params, opt_state, loss = self.policy.update_parameters(params, opt_state, data)
                    params = self.policy.update_target(params)

                # update stats
                ep_reward += reward

                timesteps += 1
                ep_timesteps += 1
                pbar.update(1)

                if done or ep_timesteps >= self.max_timesteps_per_episode:
                    print("Episode total reward:", ep_reward)
                    print("Episode total timesteps:", ep_timesteps)
                    print("--------------")

                    obs, _ = self.env.reset()
                    ep_reward = 0
                    ep_timesteps = 0


if __name__ == "__main__":
    # TODO: remove this test code after done everything
    import gymnasium as gym
    from minigrid.wrappers import FlatObsWrapper

    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    env = FlatObsWrapper(env)

    policy = DQN(
        env.observation_space.shape[0],
        env.action_space.n,
        hiddens=[128, 128, 128],
        gamma=0.99,
        learning_rate=1e-3
    )

    roller = OffPolicyRoller(policy, env)
    roller.run()
