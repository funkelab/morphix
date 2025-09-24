# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
#     "equinox",
# ]
# ///
import time
import jax
from models import ReactModel, SplitModel, Model
from utils import print_cells
from simulation import simulate, simulation_step


def run_simulation(model, num_timesteps, key):
    subkey, key = jax.random.split(key, 2)
    cells = model.initialize_cells(subkey)
    print()
    print()
    for t in range(min(num_timesteps, 10)):
        print(f"{t=}:")
        print_cells(cells)
        subkey, key = jax.random.split(key, 2)
        cells = simulation_step(cells, model, subkey)


if __name__ == "__main__":
    max_num_cells = 8
    cell_state_dims = 32
    num_timesteps = 10_000

    key = jax.random.key(1912)
    key1, key2, key3, key = jax.random.split(key, 4)
    react_model = ReactModel(cell_state_dims, cell_state_dims * 2, key=key1)
    split_model = SplitModel(cell_state_dims, cell_state_dims * 2, eps=0.0, key=key2)
    model = Model(max_num_cells, cell_state_dims, react_model, split_model, key=key3)

    # print a few iterations:
    subkey, key = jax.random.split(key, 2)
    run_simulation(model, num_timesteps, subkey)

    # benchmark many more iterations
    start = time.time()
    subkey, key = jax.random.split(key, 2)
    cells = model.initialize_cells(subkey)
    for t in range(num_timesteps):
        subkey, key = jax.random.split(key, 2)
        cells = simulation_step(cells, model, subkey)
    cells.p_split.block_until_ready()
    total = time.time() - start
    print(
        f"{num_timesteps} timesteps in {total:.3f}s "
        f"({total / num_timesteps * 1000**2:.3f}μs "
        "per iteration)"
    )

    # same with simulate function, including compilation time
    start = time.time()
    all_cells = simulate(model, num_timesteps, subkey)
    all_cells.p_split.block_until_ready()
    print(f"first run (including compilation): {time.time() - start:.3f}s")

    # and without compilation time
    start = time.time()
    all_cells = simulate(model, num_timesteps, subkey)
    all_cells.p_split.block_until_ready()
    total = time.time() - start
    print(
        f"{num_timesteps} timesteps in {total:.3f}s "
        f"({total / num_timesteps * 1000**2:.3f}μs "
        "per iteration)"
    )
