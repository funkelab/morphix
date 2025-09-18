import jax
import jax.numpy as jnp
from cell import Cell


def reward(cells: Cell, t):
    """Compute the reward of a cell state at a given point in time."""

    # placeholder: reward as few cells as possible
    num_active = (cells.parent >= 0).sum()

    # one cell for t = 0, 1; then two cells
    target = 1 + (t > 2)

    # reward is +1 for one cell, -1 otherwise
    return -1 + 2 * (num_active == target)


def trajectory_reward(cells: Cell):
    """Compute the reward of an entire trajectory."""
    num_timesteps = cells.state.shape[0]
    return jax.vmap(reward)(cells, jnp.arange(num_timesteps)).mean()


def trajectory_loss(cells: Cell):
    reward = trajectory_reward(cells)

    # p_action: (t, n)
    p_action = cells.p_split * cells.split + (1.0 - cells.p_split) * (1 - cells.split)
    log_p_action = jnp.log(p_action).sum()

    return -log_p_action * reward
