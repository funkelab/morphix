# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax",
#     "flax",
#     "optax",
#     "tqdm",
# ]
# ///
from flax import nnx
from models import ReactModel, SplitModel, Model
from simulation import simulate, simulation_step
from loss import trajectory_loss
from functools import partial
from tqdm import tqdm
from utils import print_cells
import jax
import optax


def simulation_loss(
    model_def,
    params,
    state,
    num_timesteps: int,
) -> float:
    # run the simulation
    trajectory = simulate(model_def, params, state, num_timesteps)

    # compute the loss
    return trajectory_loss(trajectory)


@partial(jax.jit, static_argnames=("num_timesteps", "optimizer"))
def train_step(
    model_def,
    params,
    state,
    num_timesteps,
    optimizer,
    opt_state,
):
    def loss_fn(params):
        return simulation_loss(model_def, params, state, num_timesteps)

    loss, grad = jax.value_and_grad(loss_fn)(params)

    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params


def create_model(max_num_cells, cell_state_dims, exploration_eps):
    rngs = nnx.Rngs(default=0, params=1, dropout=2, split_probs=3)
    react_model = ReactModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)
    split_model = SplitModel(
        cell_state_dims, cell_state_dims * 2, eps=exploration_eps, rngs=rngs
    )
    model = Model(max_num_cells, cell_state_dims, react_model, split_model, rngs=rngs)
    return model


def run_simulation(model, num_timesteps):
    model_def, params, state = model.split()
    cells = model.initialize_cells()
    # print(f"model state {state}")
    # print(f"model params {params}")
    print()
    print()
    for t in range(min(num_timesteps, 10)):
        print(f"{t=}:")
        print_cells(cells)
        cells, model_state = simulation_step(cells, model_def, params, state)


if __name__ == "__main__":
    max_num_cells = 2
    cell_state_dims = 64
    num_timesteps = 8
    num_iterations = 10_000
    learning_rate = 1e-4
    exploration_eps = 0.0

    model = create_model(max_num_cells, cell_state_dims, exploration_eps)
    cells = model.initialize_cells()

    model_def, params, state = model.split()
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    run_simulation(model, num_timesteps)

    try:
        iteration_range = tqdm(range(num_iterations))
        for i in iteration_range:
            run_simulation(model, num_timesteps)
            loss, params = train_step(
                model_def,
                params,
                state,
                num_timesteps,
                optimizer,
                opt_state,
            )
            iteration_range.set_description(f"{loss.item():.3f}")
            model = model.update_params(params)
    except KeyboardInterrupt:
        print("Training stopped early.")
        pass

    for i in range(2):
        run_simulation(model, num_timesteps)
