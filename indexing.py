# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
# ]
# ///
import jax
import jax.numpy as jnp
import time


@jax.jit
def typemask_to_indices(mask: jax.Array) -> jax.Array:
    """Compute an index array from a "type mask".

    The type mask contains n elements that mark cells:

        0   cell migrates (and should be kept as is)
        1   cell splits (two daughter cells replace it)
        2   cell is free (is not active yet)

    This is to be used on an array(-like pytree) of 3n cells with the following
    layout:

        intermediate_cells = [
            cell_1,       ..., cell_n,          # parents
            daughter_a_1, ..., daughter_a_n,    # daughters a
            daughter_b_1, ..., daughter_b_n     # daughters b
        ]

    The type mask indicates which original cells continue or split. This
    function creates ``indices``, which can be used to collect the correct
    subset of cells:

        next_cells = intermediate_cells[indices][:n]

    such that all migrating cells are kept and new daughter cells occopy the
    free slots (marked with a 2 in the type mask).
    """

    # following example for a simple mask:
    #
    # mask = [0, 0, 1, 0, 1, 2, 2, 2]

    extended_mask = jnp.concatenate(
        (
            jnp.where(mask == 1, 3, 0) + jnp.where(mask == 2, 2, 0),
            jnp.where(mask == 1, 1, 3),
            jnp.where(mask == 1, 1, 3),
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

    return indices


@jax.jit
def masks_to_indices(keep_parents: jax.Array, keep_daughters: jax.Array) -> jax.Array:
    """Compute an index array from two binary masks.

    Each mask contains n elements that mark cells:

        keep_parents:   those cells migrate (and should be kept as they are)
        keep_daughters: those cells split (two daughter cells replace each)

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

        next_cells = intermediate_cells[indices][:n]

    such that all migrating cells are kept and new daughter cells occopy the
    free slots (which are neither in keep_parents nor keep_daughters).
    """

    # following example for simple masks:
    #
    # keep_parents   = [1, 1, 0, 1, 0, 0, 0, 0]
    # keep_daughters = [0, 0, 1, 0, 1, 0, 0, 0]

    active = keep_parents | keep_daughters
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

    return indices


if __name__ == "__main__":
    keep_parents = jnp.array([1, 1, 0, 1, 0, 0, 0, 0])
    keep_daughters = jnp.array([0, 0, 1, 0, 1, 0, 0, 0])
    num_cells = len(keep_parents)
    indices = masks_to_indices(keep_parents, keep_daughters)

    num_iterations = 10_000
    start = time.time()
    for i in range(num_iterations):
        indices = masks_to_indices(keep_parents, keep_daughters)
    print(f"{(time.time() - start) / num_iterations:.7f}s")

    print(f"{keep_parents=}")
    print(f"{keep_daughters=}")
    print(f"{keep_daughters + (jnp.logical_not(keep_daughters | keep_parents)) * 2}")
    print(f"{indices=}")

    intermediate_cells = jnp.array(
        [
            # parents
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            # daughters a
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            # daughters b
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ]
    )

    cells = intermediate_cells[indices][:num_cells]

    print(f"{cells=}")
