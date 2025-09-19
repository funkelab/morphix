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
    cell_state_dims = 32
    num_timesteps = 10_000

    cells = Cell(
        position=jnp.zeros((max_num_cells, 3)),
        state=jnp.zeros((max_num_cells, cell_state_dims)),
        # initially, only one cell is active
        parent=(-jnp.ones((max_num_cells,), dtype=jnp.int16)).at[0].set(0),
        p_split=jnp.zeros((max_num_cells,)),
        split=jnp.zeros((max_num_cells,), dtype=jnp.bool),
    )

    rngs = nnx.Rngs(params=0, dropout=1, split_probs=2)
    react_model = ReactModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)
    split_model = SplitModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)
    model_def, model_state = nnx.split((react_model, split_model))

    # print a few iterations:
    for t in range(10):
        print(f"{t=}:")
        print_cells(cells)
        cells, model_state = simulation_step(
            cells, model_def, model_state, max_num_cells
        )

    # benchmark many more iterations
    start = time.time()
    for t in range(num_timesteps):
        cells, model_state = simulation_step(
            cells, model_def, model_state, max_num_cells
        )
    cells.p_split.block_until_ready()
    total = time.time() - start
    print(
        f"{num_timesteps} timesteps in {total:.3f}s "
        f"({total / num_timesteps * 1000**2:.3f}μs "
        "per iteration)"
    )

    # same with simulate function
    start = time.time()
    all_cells, model_state = simulate(cells, model_def, model_state, num_timesteps)
    all_cells.p_split.block_until_ready()
    print(f"first run (including compilation): {time.time() - start:.3f}s")
    start = time.time()
    all_cells, model_state = simulate(cells, model_def, model_state, num_timesteps)
    all_cells.p_split.block_until_ready()
    total = time.time() - start
    print(
        f"{num_timesteps} timesteps in {total:.3f}s "
        f"({total / num_timesteps * 1000**2:.3f}μs "
        "per iteration)"
    )
