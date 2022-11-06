import jax
import optax
import jax.numpy as jnp

from rl_zoo.utils.common import gather_nd, Transition
from rl_zoo.utils.model import build_mlp, hard_update
from collections import namedtuple
from typing import List, Tuple

Parameters = namedtuple("Parameters", "q q_target")
Output = namedtuple("Output", "action q")
OptimizerState = namedtuple("OptimizerState", "state")
Loss = namedtuple("Loss", "loss")


class DQN:
    """
    An implementation of Double DQN
    Reference:
        https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        https://arxiv.org/abs/1509.06461
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hiddens: List[int] = [64, 64],
                 gamma: float = 0.95,
                 learning_rate: float = 1e-3):
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.q = build_mlp(hiddens, action_dim)
        self.opt = optax.adam(learning_rate)

        self.update_parameters = jax.jit(self.update_parameters)
        self.get_action = jax.jit(self.get_action)
        self.get_random_action = jax.jit(self.get_random_action)

    def initial_parameters(self, rng) -> Parameters:
        sample_input = jnp.zeros((1, self.obs_dim))

        q_rng, q_target_rng = jax.random.split(rng, 2)

        q_params = self.q.init(q_rng, sample_input)
        q_target_params = self.q.init(q_target_rng, sample_input)

        return Parameters(q_params, q_target_params)

    def initial_optimizer(self, params: Parameters) -> OptimizerState:
        opt_state = self.opt.init(params.q)
        return OptimizerState(opt_state)

    def td_error(self, q_params, q_target_params, data: Transition):
        obs, action, reward, next_obs, done = data

        indices = jnp.stack([jnp.arange(obs.shape[0]), action], axis=1)

        q = gather_nd(self.q.apply(q_params, obs), indices)
        next_q_target = jax.lax.stop_gradient(self.q.apply(q_target_params, next_obs))
        target = reward + self.gamma * (1 - done) * jnp.max(next_q_target, axis=1)
        return jnp.mean(jnp.square(q - target))

    def update_parameters(self,
                          params: Parameters,
                          opt_state: OptimizerState,
                          data: Transition) -> Tuple[Parameters, OptimizerState, Loss]:

        grad_fn = jax.value_and_grad(self.td_error)
        loss, grads = grad_fn(params.q, params.q_target, data)
        updates, opt_state = self.opt.update(grads, opt_state.state)
        q_params = optax.apply_updates(params.q, updates)

        return (
            Parameters(q_params, params.q_target),
            OptimizerState(opt_state),
            Loss(loss)
        )

    def update_target(self, params: Parameters) -> Parameters:
        q_target_params = hard_update(params.q_target, params.q)
        return Parameters(params.q, q_target_params)

    def get_action(self, params: Parameters, obs: jnp.ndarray, rng) -> Output:
        obs = jnp.reshape(obs, (1, -1))
        q = self.q.apply(params.q, obs)
        best_action = jnp.argmax(q, axis=1)
        return Output(best_action, q)

    def get_random_action(self, params: Parameters, obs: jnp.ndarray, rng) -> Output:
        obs = jnp.reshape(obs, (1, -1))
        q = self.q.apply(params.q, obs)
        action = jax.random.randint(rng, shape=(1,), minval=0, maxval=self.action_dim)
        return Output(action, q)
