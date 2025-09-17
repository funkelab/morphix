import jax
from flax import struct


@struct.dataclass
class Cell:
    position: jax.Array
    state: jax.Array
    parent: jax.Array  # -1 if not active, otherwise index into previous state
    p_split: jax.Array
    split: jax.Array
