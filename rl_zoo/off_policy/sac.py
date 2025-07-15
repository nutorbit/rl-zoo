import jax
import optax
import jax.numpy as jnp
import haiku as hk

from rl_zoo.utils.common import Transition
from rl_zoo.utils.model import build_mlp, soft_update
from collections import namedtuple
from typing import List, Tuple, NamedTuple

QParameters = namedtuple("QParameters", "q1 q2 q1_target q2_target")
PolicyParameters = namedtuple("PolicyParameters", "policy")
AlphaParameters = namedtuple("AlphaParameters", "log_alpha")
Output = namedtuple("Output", "action q")
OptimizerState = namedtuple("OptimizerState", "q1 q2 policy alpha")
Loss = namedtuple("Loss", "q1_loss q2_loss policy_loss alpha_loss")


class Parameters(NamedTuple):
    q: QParameters
    policy: PolicyParameters
    alpha: AlphaParameters


class SAC:
    """
    An implementation of SAC (Soft Actor-Critic)
    Reference:
        https://arxiv.org/abs/1801.01290
        https://arxiv.org/abs/1812.05905
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hiddens: List[int] = [256, 256],
                 gamma: float = 0.99,
                 learning_rate: float = 3e-4,
                 tau: float = 0.005,
                 target_entropy: float = None,
                 auto_alpha: bool = True):
        self.gamma = gamma
        self.tau = tau
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.auto_alpha = auto_alpha
        
        # Set target entropy to -dim(A) if not provided
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim

        # Q networks (twin critics)
        self.q1 = build_mlp(hiddens, 1)
        self.q2 = build_mlp(hiddens, 1)
        self.q_opt = optax.adam(learning_rate)

        # Policy network (actor)
        self.policy = self._build_policy_network(hiddens)
        self.policy_opt = optax.adam(learning_rate)

        # Alpha (temperature) parameter
        self.alpha_opt = optax.adam(learning_rate)

        # JIT compile functions
        self.update_parameters = jax.jit(self.update_parameters)
        self.get_action = jax.jit(self.get_action)
        self.get_random_action = jax.jit(self.get_random_action)

    def _build_policy_network(self, hiddens: List[int]) -> hk.Transformed:
        """Build policy network that outputs mean and log_std for Gaussian policy"""
        def network(x):
            net = hk.Sequential([
                hk.nets.MLP(hiddens, activation=jax.nn.relu),
                hk.Linear(self.action_dim * 2)  # mean and log_std
            ])
            
            out = net(x)
            mean, log_std = jnp.split(out, 2, axis=-1)
            log_std = jnp.clip(log_std, -20, 2)  # Clip for stability
            
            return mean, log_std

        return hk.without_apply_rng(hk.transform(network))

    def initial_parameters(self, rng) -> Parameters:
        rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)
        
        # Q networks
        q_sample_input = jnp.zeros((1, self.obs_dim + self.action_dim))
        
        q1_params = self.q1.init(rng1, q_sample_input)
        q2_params = self.q2.init(rng2, q_sample_input)
        q1_target_params = self.q1.init(rng3, q_sample_input)
        q2_target_params = self.q2.init(rng4, q_sample_input)

        # Policy network
        policy_sample_input = jnp.zeros((1, self.obs_dim))
        policy_params = self.policy.init(rng5, policy_sample_input)

        # Alpha parameter
        log_alpha = jnp.array(0.0)

        return Parameters(
            QParameters(q1_params, q2_params, q1_target_params, q2_target_params),
            PolicyParameters(policy_params),
            AlphaParameters(log_alpha)
        )

    def initial_optimizer(self, params: Parameters) -> OptimizerState:
        q1_opt_state = self.q_opt.init(params.q.q1)
        q2_opt_state = self.q_opt.init(params.q.q2)
        policy_opt_state = self.policy_opt.init(params.policy.policy)
        alpha_opt_state = self.alpha_opt.init(params.alpha.log_alpha)

        return OptimizerState(
            q1_opt_state,
            q2_opt_state,
            policy_opt_state,
            alpha_opt_state
        )

    def _sample_action(self, mean: jnp.ndarray, log_std: jnp.ndarray, rng) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample action using reparameterization trick"""
        std = jnp.exp(log_std)
        normal = jax.random.normal(rng, mean.shape)
        
        action_raw = mean + std * normal
        action = jnp.tanh(action_raw)
        
        # Compute log probability
        log_prob = jax.scipy.stats.norm.logpdf(normal, 0, 1).sum(-1, keepdims=True)
        log_prob -= jnp.log(1 - action**2 + 1e-6).sum(-1, keepdims=True)
        
        return action, log_prob

    def q_loss(self, q_params: QParameters, policy_params: PolicyParameters, alpha_params: AlphaParameters, data: Transition, rng) -> Tuple[float, float]:
        obs, action, reward, next_obs, done = data
        
        # Current Q values
        q1 = self.q1.apply(q_params.q1, jnp.concatenate([obs, action], axis=1))
        q2 = self.q2.apply(q_params.q2, jnp.concatenate([obs, action], axis=1))

        # Target Q values
        next_mean, next_log_std = self.policy.apply(policy_params.policy, next_obs)
        next_action, next_log_prob = self._sample_action(next_mean, next_log_std, rng)
        
        next_q1_target = self.q1.apply(q_params.q1_target, jnp.concatenate([next_obs, next_action], axis=1))
        next_q2_target = self.q2.apply(q_params.q2_target, jnp.concatenate([next_obs, next_action], axis=1))
        next_q_target = jnp.minimum(next_q1_target, next_q2_target)
        
        alpha = jnp.exp(alpha_params.log_alpha)
        target = reward + self.gamma * (1 - done) * jax.lax.stop_gradient(next_q_target - alpha * next_log_prob)

        q1_loss = jnp.mean(jnp.square(q1 - target))
        q2_loss = jnp.mean(jnp.square(q2 - target))

        return q1_loss, q2_loss

    def policy_loss(self, policy_params: PolicyParameters, q_params: QParameters, alpha_params: AlphaParameters, data: Transition, rng) -> float:
        obs, _, _, _, _ = data

        mean, log_std = self.policy.apply(policy_params.policy, obs)
        action, log_prob = self._sample_action(mean, log_std, rng)

        q1 = self.q1.apply(q_params.q1, jnp.concatenate([obs, action], axis=1))
        q2 = self.q2.apply(q_params.q2, jnp.concatenate([obs, action], axis=1))
        q = jnp.minimum(q1, q2)

        alpha = jnp.exp(alpha_params.log_alpha)
        policy_loss = jnp.mean(alpha * log_prob - q)

        return policy_loss

    def alpha_loss(self, alpha_params: AlphaParameters, policy_params: PolicyParameters, data: Transition, rng) -> float:
        if not self.auto_alpha:
            return 0.0
            
        obs, _, _, _, _ = data

        mean, log_std = self.policy.apply(policy_params.policy, obs)
        _, log_prob = self._sample_action(mean, log_std, rng)

        alpha_loss = -jnp.mean(alpha_params.log_alpha * jax.lax.stop_gradient(log_prob + self.target_entropy))

        return alpha_loss

    def update_parameters(self,
                          params: Parameters,
                          opt: OptimizerState,
                          data: Transition,
                          rng) -> Tuple[Parameters, OptimizerState, Loss]:
        
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        # Update Q networks
        q_grad_fn = jax.value_and_grad(self.q_loss, has_aux=True)
        (q1_loss, q2_loss), q_grads = q_grad_fn(params.q, params.policy, params.alpha, data, rng1)
        
        q1_updates, q1_opt_state = self.q_opt.update(q_grads.q1, opt.q1)
        q2_updates, q2_opt_state = self.q_opt.update(q_grads.q2, opt.q2)
        
        new_q1_params = optax.apply_updates(params.q.q1, q1_updates)
        new_q2_params = optax.apply_updates(params.q.q2, q2_updates)

        # Update policy network
        policy_grad_fn = jax.value_and_grad(self.policy_loss)
        policy_loss, policy_grads = policy_grad_fn(params.policy, params.q, params.alpha, data, rng2)
        
        policy_updates, policy_opt_state = self.policy_opt.update(policy_grads.policy, opt.policy)
        new_policy_params = optax.apply_updates(params.policy.policy, policy_updates)

        # Update alpha
        alpha_loss = 0.0
        new_alpha_params = params.alpha.log_alpha
        alpha_opt_state = opt.alpha
        
        if self.auto_alpha:
            alpha_grad_fn = jax.value_and_grad(self.alpha_loss)
            alpha_loss, alpha_grads = alpha_grad_fn(params.alpha, params.policy, data, rng3)
            
            alpha_updates, alpha_opt_state = self.alpha_opt.update(alpha_grads.log_alpha, opt.alpha)
            new_alpha_params = optax.apply_updates(params.alpha.log_alpha, alpha_updates)

        # Update target networks
        new_q1_target_params = soft_update(params.q.q1_target, new_q1_params, self.tau)
        new_q2_target_params = soft_update(params.q.q2_target, new_q2_params, self.tau)

        return (
            Parameters(
                QParameters(new_q1_params, new_q2_params, new_q1_target_params, new_q2_target_params),
                PolicyParameters(new_policy_params),
                AlphaParameters(new_alpha_params)
            ),
            OptimizerState(q1_opt_state, q2_opt_state, policy_opt_state, alpha_opt_state),
            Loss(q1_loss, q2_loss, policy_loss, alpha_loss)
        )

    def get_action(self, params: Parameters, obs: jnp.ndarray, rng) -> Output:
        obs = jnp.reshape(obs, (1, -1))
        mean, log_std = self.policy.apply(params.policy.policy, obs)
        action, _ = self._sample_action(mean, log_std, rng)
        return Output(action, None)

    def get_random_action(self, params: Parameters, obs: jnp.ndarray, rng) -> Output:
        # For SAC, this is the same as get_action since the policy is stochastic
        return self.get_action(params, obs, rng)

    def get_deterministic_action(self, params: Parameters, obs: jnp.ndarray) -> Output:
        """Get deterministic action (mean of policy) for evaluation"""
        obs = jnp.reshape(obs, (1, -1))
        mean, _ = self.policy.apply(params.policy.policy, obs)
        action = jnp.tanh(mean)
        return Output(action, None)