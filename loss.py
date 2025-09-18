import jax
import jax.numpy as jnp
from cell import Cell


def reward(cells: Cell, t):
    """Compute the reward of a cell state at a given point in time."""

    # placeholder: reward as few cells as possible
    num_active = (cells.parent >= 0).sum()

    # one cell for t = 0, 1; then two cells
    target = 1 + (t > 2)

    # reward is negative deviation from target
    return -jnp.abs(num_active - target)


def trajectory_rewards(cells: Cell):
    """Compute the rewards of each timestep of an entire trajectory."""
    num_timesteps = cells.state.shape[0]
    return jax.vmap(reward)(cells, jnp.arange(num_timesteps))


def trajectory_loss(cells: Cell):
    rewards = trajectory_rewards(cells)

    # compute log probabilities of each action taken
    # p_action: (t, n)
    p_action = cells.p_split * cells.split + (1.0 - cells.p_split) * (1 - cells.split)
    # log_p_action: (t,)
    log_p_action = jnp.log(p_action).sum(axis=1)

    # compute "reward-to-go" for each timestep, i.e., sum of future rewards
    # rewards_to_go: (t,)
    rewards_to_go = jnp.cumsum(rewards[::-1])[::-1]

    return -(log_p_action * rewards_to_go).sum()
