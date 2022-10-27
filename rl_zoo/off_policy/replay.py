import pickle
import random
import numpy as np
import jax.numpy as jnp

from typing import *
from collections import deque


class ReplayBuffer:
    def __init__(self, size: int = 10_000):
        self.max_size = size
        self.buffer = None
        self.reset()

    def reset(self):
        self.buffer = deque(maxlen=self.max_size)

    def add_experience(
            self,
            obs: Union[jnp.ndarray, np.ndarray],
            action: int,
            reward: float,
            next_obs: Union[jnp.ndarray, np.ndarray],
            done: Any
    ):

        self.buffer.append(
            (obs, action, reward, next_obs, done)
        )

    def sample_experience(self, size: int = 32) -> Tuple[jnp.ndarray, ...]:
        obs, action, reward, next_obs, done = zip(*random.sample(self.buffer, k=min(self.current_size, size)))
        return (
            jnp.stack(obs),
            jnp.array(action),
            jnp.array(reward),
            jnp.stack(next_obs),
            jnp.array(done),
        )

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))

    @property
    def current_size(self):
        return len(self.buffer)
