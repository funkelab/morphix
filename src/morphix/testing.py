import jax
import jax.numpy as jnp
import numpy as np


def assert_equal(a, b):
    """Assert that two pytrees are equal.

    Prints more details using numpy for how the trees differ.
    """
    same = jax.tree.reduce(
        jnp.logical_and, jax.tree.map(jnp.all, jax.tree.map(jnp.equal, a, b))
    )

    if not same:

        def numpy_assert(a, b):
            return np.testing.assert_equal(np.array(a), np.array(b))

        jax.tree.map(numpy_assert, a, b)

    return same
