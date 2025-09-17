import jax
from cell import Cell


def reward(cells: Cell):
    """Compute the reward of a cell state at a given point in time."""

    # placeholder: reward as few cells as possible
    num_active = (cells.parent >= 0).sum()

    # reward is +1 for one cell, -1 otherwise
    return -1 + 2 * (num_active == 1)


def trajectory_reward(cells: Cell):
    """Compute the reward of an entire trajectory."""
    return jax.vmap(reward)(cells).mean()


def compute_loss(cells: Cell):
    reward = trajectory_reward(cells)

    # get log prob of each performed action

    # (t, n)
    cells.p_split
