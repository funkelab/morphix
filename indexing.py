# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
# ]
# ///
import jax
import jax.numpy as jnp


def masks_to_indices(
    active: jax.Array, split: jax.Array, max_num_cells: int
) -> jax.Array:
    """Compute an index array from two binary masks.

    Each mask contains n elements that mark cells:

        active: those cells are part of the simulation
        split:  those cells split (two daughter cells replace each)

    This is to be used on an array(-like pytree) of 3n cells with the following
    layout:

        intermediate_cells = [
            cell_1,       ..., cell_n,          # parents
            daughter_a_1, ..., daughter_a_n,    # daughters a
            daughter_b_1, ..., daughter_b_n     # daughters b
        ]

    The masks indicates which original cells continue or split. This function
    creates ``indices``, which can be used to collect the correct subset of
    cells:

        next_cells = intermediate_cells[indices]

    such that all migrating cells are kept and new daughter cells occopy the
    free slots (which are not active).
    """

    keep_daughters = split & active

    # following example for simple masks:
    #
    # active         = [1, 1, 1, 1, 1, 0, 0, 0]
    # split          = [0, 0, 1, 0, 1, 0, 1, 0]
    # keep_daughters = [0, 0, 1, 0, 1, 0, 0, 0]

    extended_mask = jnp.concatenate(
        (
            jnp.where(keep_daughters, 3, 0) + jnp.where(active, 0, 2),
            jnp.where(keep_daughters, 1, 3),
            jnp.where(keep_daughters, 1, 3),
        )
    )

    # extended_mask = [
    #   0, 0, *, 0, *, 2, 2, 2  # for parents
    #   *, *, 1, *, 1, *, *, *  # for daughters a
    #   *, *, 1, *, 1, *, *, *  # for daughters b
    # ]
    #
    # (* = 3, which means that this cell won't be used later)

    indices = jnp.argsort(extended_mask)

    # indices = [
    #   0, 1, 3,    # parents to keep
    #   10, 12,     # daughters a to keep
    #   18, 20,     # daughters b to keep
    #   5, 6, 7,    # free cells to keep
    #   ...         # all other indices (will be truncated later)
    # ]

    return indices[:max_num_cells]
