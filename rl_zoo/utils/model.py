import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from typing import List, Union, Tuple, Callable


def build_mlp(
    hiddens: List[int],
    output_shape: Union[int, Tuple],
    hidden_activation: Callable = jax.nn.relu
) -> hk.Transformed:

    def network(x):
        net = hk.Sequential(
            [hk.nets.MLP(hiddens, activation=hidden_activation)] + [hk.Linear(int(np.prod(output_shape)))]
        )

        out = net(x)

        if isinstance(output_shape, tuple):
            out = jnp.reshape(out, (-1, ) + output_shape)

        return out

    return hk.without_apply_rng(hk.transform(network))


@jax.jit
def soft_update(current_params, new_params, tau: float):
    return jax.tree_util.tree_map(
        lambda new, current: tau * new + (1.0 - tau) * current,
        new_params,
        current_params
    )


@jax.jit
def hard_update(current_params, new_params):
    return soft_update(current_params, new_params, 1.0)
