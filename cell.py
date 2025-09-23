import jax
import equinox as eqx


class Cell(eqx.Module):
    position: jax.Array
    state: jax.Array
    parent: jax.Array  # -1 if not active, otherwise index into previous state
    p_split: jax.Array
    split: jax.Array
