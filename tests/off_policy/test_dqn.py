import jax
import jax.numpy as jnp

from rl_zoo.off_policy.dqn import DQN
from rl_zoo.utils.common import Transition


def test_dqn():
    # initialize the model parameters
    model = DQN(11, 4)
    params = model.initial_parameters(jax.random.PRNGKey(1))
    opt_state = model.initial_optimizer(params.q)

    # generate a random transition
    obs = jax.random.uniform(jax.random.PRNGKey(1), (10, 11))
    action = jnp.zeros((10,), dtype="int32")
    reward = jax.random.uniform(jax.random.PRNGKey(1), (10,))
    next_obs = jax.random.uniform(jax.random.PRNGKey(1), (10, 11))
    done = jnp.zeros((10,), dtype="int32")

    data = Transition(obs, action, reward, next_obs, done)

    # update the parameters
    params, opt_state, loss1 = model.update_parameters(params, opt_state, data)
    params, opt_state, loss2 = model.update_parameters(params, opt_state, data)

    # check if the loss is decreasing
    assert loss1 > loss2
