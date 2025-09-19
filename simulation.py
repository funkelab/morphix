import jax.numpy as jnp
import jax
from flax import nnx
from cell import Cell
from models import ReactModel, SplitModel
from indexing import masks_to_indices
from functools import partial


def diffuse(cells: Cell) -> Cell:
    # do nothing for now
    return cells


def react(cells: Cell, react_model: ReactModel, split_model: SplitModel) -> Cell:
    # update cell state
    state = react_model(cells.state)

    # ask split model for split probabilities
    split_probs = split_model(cells.state)

    # sample an action (split or not)
    split = split_model.sample(split_probs)

    # update cells
    return cells.replace(state=state, p_split=split_probs, split=split)


def split_cell(cell: Cell):
    # do nothing for now, just copy the cell
    daughter_a = cell
    daughter_b = cell

    return (daughter_a, daughter_b)


def split_and_recombine(cells: Cell, max_num_cells: int) -> Cell:
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
    indices = masks_to_indices(active, cells.split, max_num_cells)

    # recombine
    return jax.tree.map(
        lambda p, a, b: jnp.concatenate((p, a, b))[indices],
        cells,
        daughters_a,
        daughters_b,
    )


@partial(jax.jit, static_argnames=("max_num_cells",))
def simulation_step(
    cells: Cell, model_def: nnx.GraphDef, model_state: nnx.State, max_num_cells: int
) -> Cell:
    # reassemble models
    react_model, split_model = nnx.merge(model_def, model_state)

    # perform split (as indicated from previous timestep)
    cells = split_and_recombine(cells, max_num_cells)

    # interact with the environment
    cells = diffuse(cells)

    # update the cells internally
    cells = react(cells, react_model, split_model)

    return cells, nnx.state((react_model, split_model))


@partial(jax.jit, static_argnames=("num_timesteps",))
def simulate(
    cells: Cell, model_def: nnx.GraphDef, model_state: nnx.State, num_timesteps: int
) -> Cell:
    num_cells = len(cells.parent)

    def step(carry, _):
        cells, model_state = carry
        cells, model_state = simulation_step(cells, model_def, model_state, num_cells)
        return (cells, model_state), cells

    carry = (cells, model_state)
    carry, all_cells = jax.lax.scan(step, carry, length=num_timesteps)
    _, model_state = carry

    return all_cells, model_state
