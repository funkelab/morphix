from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from .cell import Cell
from .indexing import masks_to_indices
from .models import Model, SplitModel


def diffuse(cells: Cell) -> Cell:
    # do nothing for now
    return cells


def react(
    cells: Cell,
    model: Model,
    key,
    debug: bool = False,
) -> Cell:
    key1, key2, key3 = jax.random.split(key, 3)
    num_cells = cells.state.shape[0]

    # update cell state
    state = jax.vmap(model.react_model)(cells.state)

    # update positions
    mean, std = jax.vmap(model.move_model)(cells.state)
    keys = jax.random.split(key1, num_cells)
    move, log_p_move = jax.vmap(
        lambda m, v, k: model.move_model.sample(m, v, k, return_log_p=True)
    )(mean, std, keys)
    position = cells.position + move

    if debug:
        jax.debug.print("move mean: {}", mean)
        jax.debug.print("move std: {}", std)
        jax.debug.print("move: {}", move)

    # ask split model for split probabilities
    keys = jax.random.split(key2, num_cells)
    p_split = jax.vmap(model.split_prob_model)(state, keys)

    # sample an action (split or not)
    split = model.split_prob_model.sample(p_split, key3)

    # update cells
    return cells.replace(
        log_p_move=log_p_move,
        position=position,
        state=state,
        p_split=p_split,
        split=split,
    )


def split_cell(cell: Cell, split_model: SplitModel):
    state_ratio, size_ratio, division_plane = split_model(cell.state)

    daughter_a_state = state_ratio * cell.state
    daughter_b_state = (1.0 - state_ratio) * cell.state

    daughter_a_size = size_ratio * cell.size
    daughter_b_size = (1.0 - size_ratio) * cell.size

    # compute positions of daughter cells by moving them along the division
    # plane vector, proportional to their size (i.e., the radius)
    daughter_a_position = cell.position + division_plane * daughter_a_size
    daughter_b_position = cell.position - division_plane * daughter_b_size

    daughter_a = cell.replace(
        state=daughter_a_state,
        size=daughter_a_size,
        position=daughter_a_position,
    )
    daughter_b = cell.replace(
        state=daughter_b_state,
        size=daughter_b_size,
        position=daughter_b_position,
    )

    return (daughter_a, daughter_b)


def split_and_recombine(cells: Cell, split_model: SplitModel) -> Cell:
    # total number of cells (including inactive)
    num_cells = len(cells.parent)

    # update parent indices to current indices
    # (this invalidates the parent index temporarily)
    active = cells.parent >= 0
    indices = jnp.arange(num_cells, dtype=jnp.int16)
    parent_indices = indices * active - (1 - active)
    cells = cells.replace(parent=parent_indices)

    # split all cells
    daughters_a, daughters_b = jax.vmap(split_cell, in_axes=(0, None))(
        cells, split_model
    )

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
    cells = split_and_recombine(cells, model.split_model)

    # interact with the environment
    cells = diffuse(cells)

    # update the cells internally
    cells = react(
        cells,
        model,
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
