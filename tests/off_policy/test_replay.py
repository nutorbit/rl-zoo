import jax.numpy as jnp

from rl_zoo.off_policy.replay import ReplayBuffer


def test_replay():
    rb = ReplayBuffer(2)

    n_feature = 10

    obs = jnp.ones((n_feature, ))
    action = 1
    reward = 0
    next_obs = jnp.zeros_like(obs)
    done = 1

    rb.add_experience(obs, action, reward, next_obs, done)

    assert rb.current_size == 1

    rb.add_experience(obs, action, reward, next_obs, done)
    rb.add_experience(obs, action, reward, next_obs, done)

    assert rb.current_size == 2

    obs, action, reward, next_obs, done = rb.sample_experience(2)  # (obs, action, reward, next_obs, done)

    assert obs.shape == (2, n_feature) and \
           action.shape == (2, ) and \
           reward.shape == (2, ) and \
           next_obs.shape == (2, n_feature) and \
           done.shape == (2, )
