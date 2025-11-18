import jax
import jax.numpy as jnp


def reinforcement_losses(cells, lineage_losses, gamma, entropy_regularizer):
    """Compute reinforcement losses given (non-differentiable) lineage losses."""
    # compute log probabilities of each action taken and negative entropy of
    # distribution
    #
    # p_action    : (t, n)
    # log_p_action: (t, n)
    # neg_entropy : (,)
    p_action_0 = jnp.clip(1.0 - cells.p_split, min=1e-6)
    p_action_1 = jnp.clip(cells.p_split, min=1e-6)
    log_p_action_0 = jnp.log(p_action_0)
    log_p_action_1 = jnp.log(p_action_1)
    log_p_action = log_p_action_1 * cells.split + log_p_action_0 * (1 - cells.split)
    neg_entropy = (p_action_0 * log_p_action_0 + p_action_1 * log_p_action_1).sum()

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
    # while we're at it, we also stop gradients to losses_to_go (we are only
    # interested in gradients to log_p_action for the reinforcement learning
    # part)
    #
    # losses: (t, n)
    losses = (
        log_p_action * jax.lax.stop_gradient(losses_to_go)
        + entropy_regularizer * neg_entropy
    )

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
