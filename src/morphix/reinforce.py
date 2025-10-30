import jax
import jax.numpy as jnp


def infinite_horizon_undiscounted(losses: jax.Array):
    return jnp.cumsum(losses[::-1], axis=0)[::-1]


def infinite_horizon_discounted(losses: jax.Array, gamma: float):
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

    _, discounted_values = jax.lax.scan(reverse_discount, 0.0, losses[::-1])

    return discounted_values[::-1]
