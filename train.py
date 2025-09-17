# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
#     "flax",
# ]
# ///
from flax import nnx
from models import ReactModel, SplitModel
from cell import Cell
import jax
import jax.numpy as jnp


if __name__ == "__main__":
    max_num_cells = 8
    cell_state_dims = 4

    cells = Cell(
        position=jnp.zeros((max_num_cells, 3)),
        state=jnp.zeros((max_num_cells, cell_state_dims)),
        # initially, only one cell is active
        parent=(-jnp.ones((max_num_cells,), dtype=jnp.int16)).at[0].set(0),
        p_split=jnp.ones((max_num_cells,)),
        split=jnp.zeros((max_num_cells,), dtype=jnp.bool),
    )

    rngs = nnx.Rngs(params=0, dropout=1, split_probs=3)
    react_model = ReactModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)
    split_model = SplitModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)

    print(jax.random.uniform(rngs.split_probs(), (10,)))
    print(jax.random.uniform(rngs.split_probs(), (10,)))
