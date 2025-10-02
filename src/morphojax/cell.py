import equinox as eqx
import jax


class Cell(eqx.Module):
    log_p_move: jax.Array
    position: jax.Array
    state: jax.Array
    parent: jax.Array  # -1 if not active, otherwise index into previous state
    p_split: jax.Array
    split: jax.Array
