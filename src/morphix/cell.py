import equinox as eqx
import jax


class Cell(eqx.Module):
    log_p_move: jax.Array
    position: jax.Array
    radius: jax.Array
    state: jax.Array
    parent: jax.Array  # -1 if not active, otherwise index into previous state
    p_split: jax.Array
    split: jax.Array

    def replace(
        self,
        **kwargs,
    ):
        return eqx.tree_at(
            lambda c: tuple(getattr(c, name) for name in kwargs.keys()),
            self,
            tuple(kwargs.values()),
        )
