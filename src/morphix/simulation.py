from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from .cell import Cell
from .indexing import masks_to_indices
from .models import Model, SplitModel


@partial(eqx.filter_jit)
def simulate(
    model: eqx.Module, num_timesteps: int, key, extended_attributes: bool = False
) -> Cell:
    key1, key2 = jax.random.split(key, 2)

    # create initial cells from current model
    initial_cells = model.initialize_cells(key1, extended_attributes)

    def step(cells, key):
        cells = simulation_step(cells, model, key, extended_attributes)
        return cells, cells

    keys = jax.random.split(key2, num_timesteps - 1)
    _, trajectory = jax.lax.scan(step, initial_cells, keys)

    # prepend initial cells to trajectory
    def prepend(a, i):
        return jnp.insert(a, 0, i, axis=0)

    trajectory = jax.tree.map(prepend, trajectory, initial_cells)

    return trajectory


@partial(eqx.filter_jit)
def simulation_step(
    cells: Cell, model: eqx.Module, key, extended_attributes: bool = False
) -> Cell:
    # perform split (as indicated from previous timestep)
    cells = split_and_recombine(cells, model.split_model)

    # interact with the environment
    cells = interact(cells, model, extended_attributes)

    # update the cells internally
    cells = react(
        cells,
        model,
        key,
        extended_attributes,
    )

    return cells


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


def interact(cells: Cell, model: Model, extended_attributes: bool) -> Cell:
    # secrete molecules
    cells = model.secretion_model(cells)

    # diffuse molecules
    cells = model.diffusion_model(cells)

    # sense and interact with molecules
    cells = model.sensation_model(cells)

    # mechanical update
    cells = model.mechanics_model(cells, extended_attributes)

    return cells


def react(
    cells: Cell,
    model: Model,
    key,
    extended_attributes: bool = False,
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

    # ask split model for split probabilities
    keys = jax.random.split(key2, num_cells)
    p_split = jax.vmap(model.split_prob_model)(state, keys)

    # sample an action (split or not)
    split = model.split_prob_model.sample(p_split, key3)

    # update cells
    if extended_attributes:
        return cells.replace(
            log_p_move=log_p_move,
            position=position,
            state=state,
            p_split=p_split,
            split=split,
            move=move,
        )
    else:
        return cells.replace(
            log_p_move=log_p_move,
            position=position,
            state=state,
            p_split=p_split,
            split=split,
        )


def split_cell(cell: Cell, split_model: SplitModel):
    state_ratio, volume_ratio, division_plane = split_model(cell.state)

    daughter_a_state = state_ratio * cell.state
    daughter_b_state = (1.0 - state_ratio) * cell.state

    volume = (4.0 / 3.0) * jnp.pi * cell.radius**3
    daughter_a_volume = volume_ratio * volume
    daughter_b_volume = (1.0 - volume_ratio) * volume

    daughter_a_radius = (daughter_a_volume * 3.0 / (4.0 * jnp.pi)) ** (1.0 / 3)
    daughter_b_radius = (daughter_b_volume * 3.0 / (4.0 * jnp.pi)) ** (1.0 / 3)

    # compute positions of daughter cells by moving them along the division
    # plane vector, proportional to their radius
    daughter_a_position = cell.position + division_plane * daughter_a_radius
    daughter_b_position = cell.position - division_plane * daughter_b_radius

    daughter_a = cell.replace(
        state=daughter_a_state,
        radius=daughter_a_radius,
        position=daughter_a_position,
    )
    daughter_b = cell.replace(
        state=daughter_b_state,
        radius=daughter_b_radius,
        position=daughter_b_position,
    )

    return (daughter_a, daughter_b)
