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
    def loss_fn(model_params):
        m = nnx.merge(model_def, model_state, model_params)
        m_state = nnx.state(m)
        return simulation_loss(cells, model_def, m_state, max_num_cells, num_timesteps)

    loss, grad = jax.value_and_grad(loss_fn)(model_params)

    updates, opt_state = optimizer.update(grad, opt_state, model_params)
    model_params = optax.apply_updates(model_params, updates)

    return loss, model_params


if __name__ == "__main__":
    max_num_cells = 8
    cell_state_dims = 32
    num_timesteps = 100
    num_iterations = 100_000

    cells = Cell(
        position=jnp.zeros((max_num_cells, 3)),
        state=nnx.Rngs(1912).uniform((max_num_cells, cell_state_dims)),
        # initially, only one cell is active
        parent=(-jnp.ones((max_num_cells,), dtype=jnp.int16)).at[0].set(0),
        p_split=jnp.ones((max_num_cells,)),
        split=jnp.zeros((max_num_cells,), dtype=jnp.bool),
    )

    rngs = nnx.Rngs(params=0, dropout=1, split_probs=3)
    react_model = ReactModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)
    split_model = SplitModel(cell_state_dims, cell_state_dims * 2, rngs=rngs)
    model_def, model_params, model_state = nnx.split(
        (react_model, split_model), nnx.Param, ...
    )

    optimizer = optax.adamw(learning_rate=1e-4)
    opt_state = optimizer.init(model_params)

    iteration_range = tqdm(range(num_iterations))
    for i in iteration_range:
        loss, model_params = train_step(
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
    model = nnx.merge(model_def, model_state, model_params)
    model_state = nnx.state(model)
    for t in range(10):
        print(f"{t=}:")
        print_cells(cells)
        cells, model_state = simulation_step(
            cells, model_def, model_state, max_num_cells
        )
