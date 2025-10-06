from functools import partial

import equinox as eqx
import jax
import optax

from .loss import trajectory_loss
from .simulation import simulate


def simulation_loss(
    model,
    num_timesteps: int,
    key,
    weight_position: float = 1.0,
    debug: bool = False,
) -> float:
    # run the simulation
    trajectory = simulate(model, num_timesteps, key, debug)

    # compute the loss
    return trajectory_loss(trajectory, weight_position, debug)


@partial(
    jax.jit,
    static_argnames=(
        "static",
        "num_timesteps",
        "optimizer",
        "debug",
    ),
)
def train_step(
    params,
    static,
    num_timesteps,
    optimizer,
    opt_state,
    key,
    weight_position: float = 1.0,
    debug: bool = False,
):
    @partial(jax.jit, static_argnames=("static",))
    @jax.value_and_grad
    def loss_fn(params, static, key):
        model = eqx.combine(params, static)
        return simulation_loss(model, num_timesteps, key, weight_position, debug)

    loss, grad = loss_fn(params, static, key)

    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params, opt_state
