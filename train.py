# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
#     "equinox",
#     "optax",
#     "tqdm",
# ]
# ///
from models import ReactModel, SplitModel, Model
from simulation import simulate, simulation_step
from loss import trajectory_loss
from functools import partial
from tqdm import tqdm
from utils import print_cells
import equinox as eqx
import jax
import optax


def simulation_loss(params, static, num_timesteps: int, key) -> float:
    # run the simulation
    trajectory = simulate(params, static, num_timesteps, key)

    # compute the loss
    return trajectory_loss(trajectory)


@partial(jax.jit, static_argnames=("static", "num_timesteps", "optimizer"))
def train_step(params, static, num_timesteps, optimizer, opt_state, key):
    @partial(jax.jit, static_argnames=("static",))
    def loss_fn(params, static, key):
        return simulation_loss(params, static, num_timesteps, key)

    loss, grad = jax.value_and_grad(loss_fn)(params, static, key)

    # TODO: return opt_state, we are not reusing it!
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params


def create_model(max_num_cells, cell_state_dims, exploration_eps, key):
    key1, key2, key3 = jax.random.split(key, 3)
    react_model = ReactModel(cell_state_dims, cell_state_dims * 2, key=key1)
    split_model = SplitModel(
        cell_state_dims, cell_state_dims * 2, eps=exploration_eps, key=key2
    )
    model = Model(max_num_cells, cell_state_dims, react_model, split_model, key=key3)
    return model


def run_simulation(params, static, num_timesteps, key):
    subkey, key = jax.random.split(key, 2)
    cells = model.initialize_cells(subkey)
    print()
    print()
    for t in range(min(num_timesteps, 10)):
        print(f"{t=}:")
        print_cells(cells)
        subkey, key = jax.random.split(key, 2)
        cells = simulation_step(cells, params, static, subkey)


if __name__ == "__main__":
    max_num_cells = 2
    cell_state_dims = 64
    num_timesteps = 8
    num_iterations = 10_000
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
    run_simulation(params, static, num_timesteps, simulation_key)

    try:
        iteration_range = tqdm(range(num_iterations))
        for i in iteration_range:
            # run_simulation(params, static, num_timesteps, simulation_key)
            subkey, key = jax.random.split(key, 2)
            loss, params = train_step(
                params,
                static,
                num_timesteps,
                optimizer,
                opt_state,
                subkey,
            )
            iteration_range.set_description(f"{loss.item():.3f}")
    except KeyboardInterrupt:
        print("Training stopped early.")
        pass

    model = eqx.combine(params, static)
    for i in range(2):
        run_simulation(params, static, num_timesteps, simulation_key)
