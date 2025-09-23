import jax.numpy as jnp
import jax
from flax import nnx
from cell import Cell
from models import ReactModel, SplitModel, Model
from indexing import masks_to_indices
from functools import partial


def diffuse(cells: Cell) -> Cell:
    # do nothing for now
    return cells


def react(cells: Cell, react_model: ReactModel, split_model: SplitModel) -> Cell:
    # update cell state
    state = react_model(cells.state)

    # ask split model for split probabilities
    split_probs = split_model(state)

    # sample an action (split or not)
    split = split_model.sample(split_probs)

    # update cells
    return cells.replace(state=state, p_split=split_probs, split=split)


def split_cell(cell: Cell):
    # do nothing for now, just copy the cell
    daughter_a = cell
    daughter_b = cell

    return (daughter_a, daughter_b)


def split_and_recombine(cells: Cell) -> Cell:
    # total number of cells (including inactive)
    num_cells = len(cells.parent)

    # update parent indices to current indices
    # (this invalidates the parent index temporarily)
    active = cells.parent >= 0
    indices = jnp.arange(num_cells, dtype=jnp.int16)
    parent_indices = indices * active - (1 - active)
    cells = cells.replace(parent=parent_indices)

    # split all cells
    daughters_a, daughters_b = jax.vmap(split_cell)(cells)

    # figure out which cells to keep
    indices = masks_to_indices(active, cells.split, num_cells)

    # recombine
    return jax.tree.map(
        lambda p, a, b: jnp.concatenate((p, a, b))[indices],
        cells,
        daughters_a,
        daughters_b,
    )


@partial(jax.jit)
def simulation_step(
    cells: Cell,
    # TODO: pass model here?
    model_def: nnx.GraphDef,
    params: nnx.Param,
    state: nnx.State,
) -> Cell:
    # reassemble model
    model = Model.create(model_def, params, state)

    # perform split (as indicated from previous timestep)
    cells = split_and_recombine(cells)

    # interact with the environment
    cells = diffuse(cells)

    # update the cells internally
    cells = react(cells, model.react_model, model.split_model)

    # get current model state
    _, _, state = model.split()

    return cells, state


@partial(jax.jit, static_argnames=("num_timesteps",))
def simulate(
    # TODO: pass model here?
    model_def: nnx.GraphDef,
    params: nnx.Param,
    state: nnx.State,
    num_timesteps: int,
) -> Cell:
    # reassemble model
    model = Model.create(model_def, params, state)

    # create initial cells from current model and params
    initial_cells = model.initialize_cells()

    def step(carry, _):
        cells, state = carry
        cells, state = simulation_step(cells, model_def, params, state)
        return (cells, state), cells

    carry = (initial_cells, state)
    _, trajectory = jax.lax.scan(step, carry, length=num_timesteps - 1)

    # prepend initial cells to trajectory
    def prepend(a, i):
        return jnp.insert(a, 0, i, axis=0)

    trajectory = jax.tree.map(prepend, trajectory, initial_cells)

    return trajectory
