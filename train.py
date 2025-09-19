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
from models import ReactModel, SplitModel
from cell import Cell
from simulation import simulate, simulation_step
from loss import trajectory_loss
from functools import partial
from tqdm import tqdm
from utils import print_cells
import jax
import jax.numpy as jnp
import optax


def simulation_loss(
    initial_cells: Cell,
    model_def: nnx.GraphDef,
    model_state: nnx.State,
    max_num_cells: int,
    num_timesteps: int,
) -> float:
    # run the simulation
    trajectory, model_state = simulate(
        initial_cells, model_def, model_state, num_timesteps
    )

    # compute the loss
    return trajectory_loss(trajectory)


@partial(jax.jit, static_argnames=("max_num_cells", "num_timesteps", "optimizer"))
def train_step(
    cells,
    model_def,
    model_state,
    model_params,
    max_num_cells,
    num_timesteps,
    optimizer,
    opt_state,
):
    def loss_fn(learnable_params):
        model_params, initial_cell_states = learnable_params
        m = nnx.merge(model_def, model_state, model_params)
        m_state = nnx.state(m)
        c = cells.replace(state=initial_cell_states)
        return simulation_loss(c, model_def, m_state, max_num_cells, num_timesteps)

    loss, grad = jax.value_and_grad(loss_fn)((model_params, cells.state))

    updates, opt_state = optimizer.update(grad, opt_state, (model_params, cells.state))
    model_params, initial_cell_states = optax.apply_updates(
        (model_params, cells.state), updates
    )

    cells = cells.replace(state=initial_cell_states)

    return loss, model_params, cells


if __name__ == "__main__":
    max_num_cells = 8
    cell_state_dims = 32
    num_timesteps = 20
    num_iterations = 100_000
    learning_rate = 1e-6
    exploration_eps = 0.1

    cells = Cell(
        position=jnp.zeros((max_num_cells, 3)),
        state=nnx.Rngs(1912).uniform((max_num_cells, cell_state_dims)),
        # initially, only one cell is active
        parent=(-jnp.ones((max_num_cells,), dtype=jnp.int16)).at[0].set(0),
        p_split=jnp.ones((max_num_cells,)) * 1e-4,
        split=jnp.zeros((max_num_cells,), dtype=jnp.bool),
    )

    rngs = nnx.Rngs(params=0, dropout=1, split_probs=3)
    react_model = ReactModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)
    split_model = SplitModel(
        cell_state_dims, cell_state_dims * 2, eps=exploration_eps, rngs=rngs
    )
    model_def, model_params, model_state = nnx.split(
        (react_model, split_model), nnx.Param, ...
    )

    # we want to optimize both the model params and the inital cell state
    learnable_params = (model_params, cells.state)

    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(learnable_params)

    # print a few iterations (new variables to avoid overwriting the inital
    # state)
    model = nnx.merge(model_def, model_state, model_params)
    m = nnx.state(model)
    c = cells
    for t in range(10):
        print(f"{t=}:")
        print_cells(c)
        c, m = simulation_step(c, model_def, m, max_num_cells)

    print(f"Initial model state {model_state}")
    print(f"Initial model params {model_params}")

    iteration_range = tqdm(range(num_iterations))
    for i in iteration_range:
        loss, model_params, cells = train_step(
            cells,
            model_def,
            model_state,
            model_params,
            max_num_cells,
            num_timesteps,
            optimizer,
            opt_state,
        )
        iteration_range.set_description(f"{loss=}")
        if i % 1000 == 0:
            print(f"iteration {i}: loss {loss}")

    # print a few iterations:
    split_model = SplitModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)
    model_def, _, model_state = nnx.split((react_model, split_model), nnx.Param, ...)
    model = nnx.merge(model_def, model_state, model_params)
    model_state = nnx.state(model)
    print(f"Final model state {model_state}")
    print(f"Final model params {model_params}")
    for t in range(10):
        print(f"{t=}:")
        print_cells(cells)
        cells, model_state = simulation_step(
            cells, model_def, model_state, max_num_cells
        )
