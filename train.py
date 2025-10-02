# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
#     "equinox",
#     "optax",
#     "tqdm",
# ]
# ///
from models import create_model
from simulation import simulate, simulation_step
from loss import trajectory_loss
from tqdm import tqdm
from utils import print_cells
from functools import partial
import jax.numpy as jnp
import equinox as eqx
import jax
import optax


def simulation_loss(
    model,
    num_timesteps: int,
    key,
    weight_position: float = 1.0,
    debug: bool = False,
) -> float:
    # run the simulation
    trajectory = simulate(model, num_timesteps, key, debug)

    # compute the loss
    return trajectory_loss(trajectory, weight_position, debug)


@partial(
    jax.jit,
    static_argnames=(
        "static",
        "num_timesteps",
        "optimizer",
        "debug",
    ),
)
def train_step(
    params,
    static,
    num_timesteps,
    optimizer,
    opt_state,
    key,
    weight_position: float = 1.0,
    debug: bool = False,
):
    @partial(jax.jit, static_argnames=("static",))
    @jax.value_and_grad
    def loss_fn(params, static, key):
        model = eqx.combine(params, static)
        return simulation_loss(model, num_timesteps, key, weight_position, debug)

    loss, grad = loss_fn(params, static, key)

    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params, opt_state


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
    max_num_cells = 4
    cell_state_dims = 64
    num_timesteps = 8
    num_iterations = 100_000
    learning_rate = 1e-4
    exploration_eps = 0.0

    key = jax.random.key(1912)
    simulation_key = jax.random.key(1954)

    # create model
    subkey, key = jax.random.split(key, 2)
    model = create_model(max_num_cells, cell_state_dims, exploration_eps, subkey)
    params, static = model.partition()
    subkey, key = jax.random.split(key, 2)
    cells = model.initialize_cells(subkey)

    # create optimizer for params
    print(params)
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    # run a first simulation
    run_simulation(model, num_timesteps, simulation_key)

    try:
        iteration_range = tqdm(range(num_iterations))
        for i in iteration_range:
            subkey, key = jax.random.split(key, 2)
            loss, params, opt_state = train_step(
                params,
                static,
                num_timesteps,
                optimizer,
                opt_state,
                subkey,
                weight_position=0.01,
                debug=False,
            )
            iteration_range.set_description(f"{loss.item():.3f}")
            assert not jnp.isnan(loss)
    except KeyboardInterrupt:
        print("Training stopped early.")
        pass

    model = eqx.combine(params, static)
    for i in range(2):
        run_simulation(model, num_timesteps, simulation_key)
