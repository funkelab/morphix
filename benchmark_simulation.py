# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
#     "flax",
# ]
# ///
import jax.numpy as jnp
import time
from flax import nnx
from cell import Cell
from models import ReactModel, SplitModel
from utils import print_cells
from simulation import simulate, simulation_step


if __name__ == "__main__":
    max_num_cells = 8
    cell_state_dims = 4

    cells = Cell(
        position=jnp.zeros((max_num_cells, 3)),
        state=jnp.zeros((max_num_cells, cell_state_dims)),
        # initially, only one cell is active
        parent=(-jnp.ones((max_num_cells,), dtype=jnp.int16)).at[0].set(0),
        p_split=jnp.zeros((max_num_cells,)),
        split=jnp.zeros((max_num_cells,), dtype=jnp.bool),
    )

    rngs = nnx.Rngs(params=0, dropout=1, split_probs=3)
    react_model = ReactModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)
    split_model = SplitModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)

    # print a few iterations:
    for t in range(4):
        print(f"{t=}:")
        print_cells(cells)
        cells = simulation_step(cells, react_model, split_model, max_num_cells)

    # benchmark many more iterations
    num_iterations = 10_000
    start = time.time()
    for t in range(num_iterations):
        cells = simulation_step(cells, react_model, split_model, max_num_cells)
    total = time.time() - start
    print(
        f"{num_iterations} timesteps in {total:.3f}s ({total / num_iterations:.5f}s per iteration)"
    )

    # same with simulate function
    start = time.time()
    all_cells = simulate(cells, react_model, split_model, num_iterations)
    total = time.time() - start
    print(
        f"{num_iterations} timesteps in {total:.3f}s ({total / num_iterations:.5f}s per iteration)"
    )
