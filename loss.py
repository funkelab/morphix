import jax
import jax.numpy as jnp
from cell import Cell


def reward(cells: Cell, t):
    """Compute the reward of a cell state at a given point in time."""

    active = cells.parent >= 0

    # placeholder: reward as few cells as possible
    num_active = active.sum()

    # one cell until t=3; then two cells
    target = 1 + (t > 3)

    diff = num_active - target

    # reward is negative deviation from target, but only for active cells
    rewards = active * -jnp.abs(diff)

    # (n,)
    return rewards


def trajectory_rewards(cells: Cell):
    """Compute the rewards of each timestep of an entire trajectory."""
    num_timesteps = cells.state.shape[0]
    return jax.vmap(reward)(cells, jnp.arange(num_timesteps))


def trajectory_loss(cells: Cell, debug=False):
    # (t, n)
    rewards = trajectory_rewards(cells)

    # compute log probabilities of each action taken
    # p_action: (t, n)
    p_action = cells.p_split * cells.split + (1.0 - cells.p_split) * (1 - cells.split)
    # log_p_action: (t, n)
    log_p_action = jnp.log(p_action)

    # compute "reward-to-go" for each timestep and cell, i.e., sum of future
    # rewards rewards_to_go: (t, n)
    rewards_to_go = jnp.cumsum(rewards[::-1], axis=0)[::-1]

    if debug:
        jax.debug.print("rewards: {}", rewards)
        jax.debug.print("split: {}", cells.split)
        jax.debug.print("p_split: {}", cells.p_split)
        jax.debug.print("p_action: {}", p_action)
        jax.debug.print("log_p_action: {}", log_p_action)
        jax.debug.print("rewards_to_go {}", rewards_to_go)

    # zero-out rewards for inactive cells
    active = cells.parent >= 0
    rewards_to_go = rewards_to_go * active

    # per-cell losses: (t, n)
    losses = -log_p_action * rewards_to_go

    if debug:
        jax.debug.print("rewards_to_go * active {}", rewards_to_go)
        jax.debug.print("per-cell losses: {}", losses)

    return losses.sum()
