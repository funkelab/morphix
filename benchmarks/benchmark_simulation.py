# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "morphix",
# ]
#
# [tool.uv.sources]
# morphix = { path = "../", editable = true }
# ///
import time

import jax
import morphix as mx

if __name__ == "__main__":
    max_num_cells = 8
    cell_state_dims = 32
    num_molecules = 8
    delta_t = 1.0
    num_timesteps = 10_000

    key = jax.random.key(1912)
    subkey, key = jax.random.split(key, 2)
    model = mx.create_model(key, max_num_cells, cell_state_dims, num_molecules, delta_t)

    # print a few iterations:
    subkey, key = jax.random.split(key, 2)
    mx.print_simulation(model, num_timesteps=10, key=subkey)

    # benchmark many more iterations
    start = time.time()
    subkey, key = jax.random.split(key, 2)
    cells = model.initialize_cells(subkey)
    for t in range(num_timesteps):
        subkey, key = jax.random.split(key, 2)
        cells = mx.simulation_step(cells, model, subkey)
    cells.p_split.block_until_ready()
    total = time.time() - start
    print(
        f"{num_timesteps} timesteps in {total:.3f}s "
        f"({total / num_timesteps * 1000**2:.3f}μs "
        "per iteration)"
    )

    # same with simulate function, including compilation time
    start = time.time()
    all_cells = mx.simulate(model, num_timesteps, subkey)
    all_cells.p_split.block_until_ready()
    print(f"first run (including compilation): {time.time() - start:.3f}s")

    # and without compilation time
    start = time.time()
    all_cells = mx.simulate(model, num_timesteps, subkey)
    all_cells.p_split.block_until_ready()
    total = time.time() - start
    print(
        f"{num_timesteps} timesteps in {total:.3f}s "
        f"({total / num_timesteps * 1000**2:.3f}μs "
        "per iteration)"
    )
