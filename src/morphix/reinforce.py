import jax
import jax.numpy as jnp


def reinforcement_losses(cells, lineage_losses, gamma):
    """Compute reinforcement losses given (non-differentiable) lineage losses."""
    # compute log probabilities of each action taken
    #
    # log_p_action: (t, n)
    log_p_action = jnp.log(
        cells.p_split * cells.split + (1.0 - cells.p_split) * (1 - cells.split)
    )

    # compute "losses-to-go" for each timestep, i.e., sum of future losses
    #
    # lineage_losses: (t, n) or (t, 1)
    # losses_to_go  : (t, n) or (t, 1)
    if gamma == 1.0:
        losses_to_go = infinite_horizon_undiscounted(lineage_losses)
    else:
        losses_to_go = infinite_horizon_discounted(lineage_losses, gamma=gamma)

    # compute per-cell reinforcement losses by broadcasting losses-to-go over
    # all cells (this includes inactive cells, which will be zeroed-out later)
    #
    # losses: (t, n)
    losses = log_p_action * losses_to_go

    return losses


def infinite_horizon_undiscounted(losses: jax.Array):
    """Given losses over time, compute the per-timestep future loss."""
    return jnp.cumsum(losses[::-1], axis=0)[::-1]


def infinite_horizon_discounted(losses: jax.Array, gamma: float):
    """Given losses over time, compute the per-timestep future loss with a decay."""
    # discounted values are:
    #
    # d_n = l_n
    # d_i = g^0 * l_i + g^1 * l_i+1 + ...
    #     = d_i+1 * g + l_i
    #
    # (where g is gamma and l the losses)
    #
    # so we can compute d_i from the next d_i+1 by scanning in reverse order
    # over the losses

    def reverse_discount(next_d, current_loss):
        d = next_d * gamma + current_loss
        return d, d

    _, discounted_values = jax.lax.scan(
        reverse_discount, jnp.zeros_like(losses[0], dtype=jnp.float32), losses[::-1]
    )

    return discounted_values[::-1]
