from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from .cell import Cell
from .indexing import masks_to_indices
from .models import MoveModel, ReactModel, SplitModel


def diffuse(cells: Cell) -> Cell:
    # do nothing for now
    return cells


def react(
    cells: Cell,
    react_model: ReactModel,
    move_model: MoveModel,
    split_model: SplitModel,
    key,
    debug: bool = False,
) -> Cell:
    key1, key2, key3 = jax.random.split(key, 3)
    num_cells = cells.state.shape[0]

    # update cell state
    state = jax.vmap(react_model)(cells.state)

    # update positions
    mean, std = jax.vmap(move_model)(cells.state)
    keys = jax.random.split(key1, num_cells)
    move, log_p_move = jax.vmap(
        lambda m, v, k: move_model.sample(m, v, k, return_log_p=True)
    )(mean, std, keys)
    position = cells.position + move

    if debug:
        jax.debug.print("move mean: {}", mean)
        jax.debug.print("move std: {}", std)
        jax.debug.print("move: {}", move)

    # ask split model for split probabilities
    keys = jax.random.split(key2, num_cells)
    p_split = jax.vmap(split_model)(state, keys)

    # sample an action (split or not)
    split = split_model.sample(p_split, key3)

    # update cells
    return eqx.tree_at(
        lambda c: (c.log_p_move, c.position, c.state, c.p_split, c.split),
        cells,
        (log_p_move, position, state, p_split, split),
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


@partial(eqx.filter_jit)
def simulation_step(cells: Cell, model: eqx.Module, key, debug: bool = False) -> Cell:
    # perform split (as indicated from previous timestep)
    cells = split_and_recombine(cells)

    # interact with the environment
    cells = diffuse(cells)

    # update the cells internally
    cells = react(
        cells,
        model.react_model,
        model.move_model,
        model.split_model,
        key,
        debug,
    )

    return cells


@partial(eqx.filter_jit)
def simulate(model: eqx.Module, num_timesteps: int, key, debug: bool = False) -> Cell:
    key1, key2 = jax.random.split(key, 2)

    # create initial cells from current model
    initial_cells = model.initialize_cells(key1)

    def step(cells, key):
        cells = simulation_step(cells, model, key, debug)
        return cells, cells

    keys = jax.random.split(key2, num_timesteps - 1)
    _, trajectory = jax.lax.scan(step, initial_cells, keys)

    # prepend initial cells to trajectory
    def prepend(a, i):
        return jnp.insert(a, 0, i, axis=0)

    trajectory = jax.tree.map(prepend, trajectory, initial_cells)

    return trajectory
