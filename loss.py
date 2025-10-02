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
    # lineage reward is in [0, num_cells]
    rewards_lineage = active * -jnp.abs(diff)

    # add position reward (y coordinate should be t) in (0.0, 1.0]
    rewards_position = (
        active * 0.01 / jnp.clip((cells.position[:, 1] - t) ** 2, a_min=1e-2)
    )

    # (n,)
    return rewards_lineage, rewards_position


def trajectory_rewards(cells: Cell):
    """Compute the rewards of each timestep of an entire trajectory."""
    num_timesteps = cells.state.shape[0]
    return jax.vmap(reward)(cells, jnp.arange(num_timesteps))


def trajectory_loss(cells: Cell, weight_position: float = 1.0, debug: bool = False):
    # (t, n)
    rewards_lineage, rewards_position = trajectory_rewards(cells)

    # compute log probabilities of each action taken
    # p_action: (t, n)
    p_action = cells.p_split * cells.split + (1.0 - cells.p_split) * (1 - cells.split)
    # log_p_action: (t, n)
    log_p_action = jnp.log(p_action)

    # add log prob of movement
    log_p_move = cells.log_p_move

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
        jax.debug.print("log_p_move: {}", log_p_move)

    # zero-out rewards for inactive cells
    active = cells.parent >= 0
    rewards_lineage_to_go = rewards_lineage_to_go * active
    rewards_position_to_go = rewards_position_to_go * active

    # per-cell losses: (t, n)
    losses_lineage = -log_p_action * rewards_lineage_to_go
    losses_position = -log_p_move * rewards_position_to_go
    losses = losses_lineage + weight_position * losses_position

    if debug:
        jax.debug.print("per-cell losses: {}", losses)

    return losses.sum()
