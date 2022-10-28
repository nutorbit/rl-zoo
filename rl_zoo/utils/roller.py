import jax
import numpy as np

from tqdm import tqdm
from typing import Any
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
                 eval_every: int = 500,
                 batch_size: int = 32,
                 logger: Any = None,
                 ):

        self.policy = policy
        self.env = env
        self.warm_start = warm_start
        self.max_timesteps = max_timesteps
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.epsilon = epsilon
        self.replay_size = replay_size
        self.update_every = update_every
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.logger = logger

    def eval(self, params):
        rng = jax.random.PRNGKey(1)

        obs, _ = self.env.reset()
        done = False
        ep_reward = 0
        ep_timesteps = 0
        while not done:

            rng, rng_action = jax.random.split(rng)
            action = self.policy.get_action(params, obs, rng_action)
            if not isinstance(action.action, np.ndarray):
                action = np.array(action.action)[0]

            next_obs, reward, done, _, _ = self.env.step(action)
            obs = next_obs
            ep_reward += reward
            ep_timesteps += 1

            if ep_timesteps >= self.max_timesteps_per_episode:
                break

        if self.logger is not None:
            self.logger.log({"Evaluation returns": ep_reward})

    def run(self):
        rb = ReplayBuffer(self.replay_size)

        rng = jax.random.PRNGKey(1)
        rng, rng_init = jax.random.split(rng)
        params = self.policy.initial_parameters(rng_init)
        opt_state = self.policy.initial_optimizer(params)

        obs, _ = self.env.reset()

        timesteps = 0
        ep_reward = 0
        ep_timesteps = 0

        with tqdm(total=self.max_timesteps) as pbar:
            while timesteps < self.max_timesteps:
                # select action
                rng, rng_action = jax.random.split(rng)
                if timesteps <= self.warm_start or jax.random.uniform(rng_action) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy.get_action(params, obs, rng_action)
                    if not isinstance(action.action, np.ndarray):
                        action = np.array(action.action)[0]

                # step
                next_obs, reward, done, _, _ = self.env.step(action)
                rb.add_experience(obs, action, reward, next_obs, done)
                obs = next_obs

                if timesteps % self.update_every == 0 and rb.current_size >= self.batch_size:
                    data = rb.sample_experience(self.batch_size)
                    params, opt_state, loss = self.policy.update_parameters(params, opt_state, data)

                    if self.logger is not None:
                        self.logger.log(loss._asdict())

                # update stats
                ep_reward += reward

                timesteps += 1
                ep_timesteps += 1
                pbar.update(1)

                if timesteps % self.eval_every == 0:
                    self.eval(params)
                    obs, _ = self.env.reset()

                if done or ep_timesteps >= self.max_timesteps_per_episode:
                    if self.logger is not None:
                        self.logger.log({
                            "Episode returns": ep_reward,
                            "Episode total timesteps": ep_timesteps,
                        })

                    obs, _ = self.env.reset()
                    ep_reward = 0
                    ep_timesteps = 0

                    params = self.policy.update_target(params)
