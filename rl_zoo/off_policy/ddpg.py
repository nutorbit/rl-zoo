import jax
import optax
import jax.numpy as jnp

from rl_zoo.utils.common import Transition
from rl_zoo.utils.model import build_mlp, hard_update
from collections import namedtuple
from typing import List, Tuple, NamedTuple, Any

QParameters = namedtuple("QParameters", "q q_target")
PolicyParameters = namedtuple("PolicyParameters", "policy")
Output = namedtuple("Output", "action q")
OptimizerState = namedtuple("OptimizerState", "q policy")
Loss = namedtuple("Loss", "q_loss policy_loss")


class Parameters(NamedTuple):
    q: QParameters
    policy: PolicyParameters


class DDPG:
    """
    An implementation of DDPG
    Reference:
        https://arxiv.org/abs/1509.02971
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

        # q network
        self.q = build_mlp(hiddens, action_dim)
        self.q_opt = optax.adam(learning_rate)

        # policy network
        self.policy = build_mlp(hiddens, action_dim)
        self.policy_opt = optax.adam(learning_rate)

        # self.update_parameters = self.update_parameters
        # self.get_action = self.get_action

        self.update_parameters = jax.jit(self.update_parameters)
        self.get_action = jax.jit(self.get_action)
        self.get_random_action = jax.jit(self.get_random_action)

    def initial_parameters(self, rng) -> Parameters:
        # q network
        sample_input = jnp.zeros((1, self.obs_dim + self.action_dim))

        q_rng, q_target_rng = jax.random.split(rng, 2)
        q_params = self.q.init(q_rng, sample_input)
        q_target_params = self.q.init(q_target_rng, sample_input)

        # policy network
        sample_input = jnp.zeros((1, self.obs_dim))

        policy_params = self.policy.init(rng, sample_input)

        return Parameters(
            QParameters(q_params, q_target_params),
            PolicyParameters(policy_params)
        )

    def initial_optimizer(self, params: Parameters) -> OptimizerState:
        # q network
        opt_state = self.q_opt.init(params.q.q)

        # policy network
        policy_opt_state = self.policy_opt.init(params.policy.policy)

        return OptimizerState(
            opt_state,
            policy_opt_state
        )

    def q_loss(self, q_params: QParameters, policy_params: PolicyParameters, data: Transition) -> float:
        obs, action, reward, next_obs, done = data

        q = self.q.apply(q_params.q, jnp.concatenate([obs, action], axis=1))

        next_action = self.policy.apply(policy_params.policy, next_obs)
        next_q_target = self.q.apply(q_params.q_target, jnp.concatenate([next_obs, next_action], axis=1))

        target = reward + self.gamma * (1 - done) * jax.lax.stop_gradient(next_q_target)

        return jnp.mean(jnp.square(q - target))

    def policy_loss(self, policy_params: PolicyParameters, q_params: QParameters, data: Transition) -> float:
        obs, _, _, _, _ = data

        action = self.policy.apply(policy_params.policy, obs)
        q = self.q.apply(q_params.q, jnp.concatenate([obs, action], axis=1))

        return -jnp.mean(q)

    def update_parameters(self,
                          params: Parameters,
                          opt: OptimizerState,
                          data: Transition) -> Tuple[Parameters, OptimizerState, Loss]:

        # update q network
        q_grad_fn = jax.value_and_grad(self.q_loss)
        q_loss, q_grads = q_grad_fn(params.q, params.policy, data)  # q_grads is for both q and q_target
        q_updates, q_opt_state = self.q_opt.update(q_grads.q, opt.q)
        q_params = optax.apply_updates(params.q.q, q_updates)

        new_q_params = QParameters(q_params, params.q.q_target)

        # update policy network
        policy_grad_fn = jax.value_and_grad(self.policy_loss)
        policy_loss, policy_grads = policy_grad_fn(params.policy, params.q, data)
        policy_updates, policy_opt_state = self.policy_opt.update(policy_grads.policy, opt.policy)
        policy_params = optax.apply_updates(params.policy.policy, policy_updates)

        new_policy_params = PolicyParameters(policy_params)

        return (
            Parameters(new_q_params, new_policy_params),
            OptimizerState(q_opt_state, policy_opt_state),
            Loss(q_loss, policy_loss)
        )

    def update_target(self, params: Parameters) -> Parameters:
        q_target_params = hard_update(params.q.q_target, params.q.q)
        return Parameters(
            QParameters(params.q.q, q_target_params),
            params.policy
        )

    def get_action(self, params: Parameters, obs: jnp.ndarray, rng) -> Output:
        obs = jnp.reshape(obs, (1, -1))
        return Output(
            self.policy.apply(params.policy.policy, obs),
            None
        )

    def get_random_action(self, params: Parameters, obs: jnp.ndarray, rng) -> Output:
        obs = jnp.reshape(obs, (1, -1))
        action = self.policy.apply(params.policy.policy, obs)
        return Output(
            action + jax.random.normal(rng, action.shape),
            None
        )
