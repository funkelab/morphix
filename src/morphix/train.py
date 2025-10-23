from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .cell import Cell
from .simulation import simulate


@partial(
    jax.jit,
    static_argnames=(
        "static",
        "reward_function",
        "num_timesteps",
        "optimizer",
        "debug",
    ),
)
def train_step(
    params,
    static,
    reward_function,
    num_timesteps,
    optimizer,
    opt_state,
    key,
    debug: bool = False,
):
    """Perform a single training step."""

    @partial(jax.jit, static_argnames=("static",))
    @jax.value_and_grad
    def loss_fn(params, static, key):
        model = eqx.combine(params, static)
        return simulation_loss(model, reward_function, num_timesteps, key, debug)

    loss, grad = loss_fn(params, static, key)

    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params, opt_state


def simulation_loss(
    model,
    reward_function,
    num_timesteps: int,
    key,
    debug: bool = False,
) -> float:
    """Run the simulation and compute the loss for the given reward function."""
    # run the simulation
    trajectory = simulate(model, num_timesteps, key, debug)

    # compute the loss
    return trajectory_loss(
        trajectory,
        reward_function,
        model.delta_t,
        debug,
    )


def trajectory_loss(
    cells: Cell,
    reward_function,
    delta_t,
    debug: bool = False,
):
    """Compute the loss for an entire trajectory."""
    # (t, n)
    rewards_lineage, rewards_position = trajectory_rewards(
        cells, reward_function, delta_t
    )

    # compute log probabilities of each action taken
    # p_action: (t, n)
    p_action = cells.p_split * cells.split + (1.0 - cells.p_split) * (1 - cells.split)
    # log_p_action: (t, n)
    log_p_action = jnp.log(p_action)

    # add log prob of motility
    log_p_motility = cells.log_p_motility

    # compute "reward-to-go" for each timestep and cell, i.e., sum of future
    # rewards rewards_to_go: (t, n)
    rewards_lineage_to_go = jnp.cumsum(rewards_lineage[::-1], axis=0)[::-1]
    rewards_position_to_go = jnp.cumsum(rewards_position[::-1], axis=0)[::-1]

    if debug:
        jax.debug.print("rewards_lineage: {}", rewards_lineage)
        jax.debug.print("rewards_position: {}", rewards_position)
        jax.debug.print("split: {}", cells.split)
        jax.debug.print("p_split: {}", cells.p_split)
        jax.debug.print("p_action: {}", p_action)
        jax.debug.print("log_p_action: {}", log_p_action)
        jax.debug.print("log_p_motility: {}", log_p_motility)

    # zero-out rewards for inactive cells
    active = cells.active
    rewards_lineage_to_go = rewards_lineage_to_go * active
    rewards_position_to_go = rewards_position_to_go * active

    # per-cell losses: (t, n)
    losses_lineage = -log_p_action * rewards_lineage_to_go
    losses_position = -log_p_motility * rewards_position_to_go
    losses = losses_lineage + losses_position

    if debug:
        jax.debug.print("per-cell losses: {}", losses)

    # mean loss of active cells
    return losses.sum() / active.sum()


def trajectory_rewards(cells: Cell, reward_function, delta_t):
    """Compute the rewards of each timestep of an entire trajectory."""
    num_timesteps = cells.state.shape[0]
    return jax.vmap(reward_function)(cells, jnp.arange(num_timesteps) * delta_t)
