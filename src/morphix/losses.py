import jax.numpy as jnp

from .cell import Cell


def pitchfork_reward(cells: Cell, t):
    """A reward function that gives rise to a "pitchfork" lineage.

    For testing purposes.
    """
    active = cells.active

    # get the number of active cells
    num_active = active.sum()

    # one cell until t=3; then two cells
    target = 1 + (t > 3)

    diff = num_active - target

    # reward is negative deviation from target, but only for active cells
    # lineage reward is in [0, num_cells]
    rewards_lineage = active * -jnp.abs(diff)

    # add position reward (y coordinate should be t) in (0.0, 1.0]
    rewards_position = (
        active * 0.01 / jnp.clip((cells.position[:, 1] - t) ** 2, a_min=1e-2)
    )

    # (n,)
    return rewards_lineage, rewards_position
