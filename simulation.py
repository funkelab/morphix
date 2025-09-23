import jax.numpy as jnp
import jax
import equinox as eqx
from cell import Cell
from models import ReactModel, SplitModel
from indexing import masks_to_indices
from functools import partial


def diffuse(cells: Cell) -> Cell:
    # do nothing for now
    return cells


def react(cells: Cell, react_model: ReactModel, split_model: SplitModel, key) -> Cell:
    key1, key2 = jax.random.split(key)

    # update cell state
    state = jax.vmap(react_model)(cells.state)

    # ask split model for split probabilities
    keys = jax.random.split(key1, cells.state.shape[0])
    p_split = jax.vmap(split_model)(state, keys)

    # sample an action (split or not)
    split = split_model.sample(p_split, key2)

    # update cells
    return eqx.tree_at(
        lambda c: (c.state, c.p_split, c.split), cells, (state, p_split, split)
    )


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
    cells = eqx.tree_at(lambda c: c.parent, cells, parent_indices)

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


@partial(jax.jit, static_argnames=("static",))
def simulation_step(cells: Cell, params: eqx.Module, static: eqx.Module, key) -> Cell:
    # perform split (as indicated from previous timestep)
    cells = split_and_recombine(cells)

    # interact with the environment
    cells = diffuse(cells)

    # update the cells internally
    model = eqx.combine(params, static)
    cells = react(cells, model.react_model, model.split_model, key)

    return cells


@partial(
    jax.jit,
    static_argnames=(
        "static",
        "num_timesteps",
    ),
)
def simulate(params: eqx.Module, static: eqx.Module, num_timesteps: int, key) -> Cell:
    key1, key2 = jax.random.split(key, 2)

    # create initial cells from current model and params
    model = eqx.combine(params, static)
    initial_cells = model.initialize_cells(key1)

    def step(cells, key):
        cells = simulation_step(cells, params, static, key)
        return cells, cells

    keys = jax.random.split(key2, num_timesteps - 1)
    _, trajectory = jax.lax.scan(step, initial_cells, keys)

    # prepend initial cells to trajectory
    def prepend(a, i):
        return jnp.insert(a, 0, i, axis=0)

    trajectory = jax.tree.map(prepend, trajectory, initial_cells)

    return trajectory
