from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from .cell import Cell
from .indexing import masks_to_indices
from .models import Model, SplitModel


@partial(eqx.filter_jit)
def simulate(
    model: Model, num_timesteps: int, key, extended_attributes: bool = False
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
    cells: Cell, model: Model, key, extended_attributes: bool = False
) -> Cell:
    # perform split (as indicated from previous timestep)
    cells = split_and_recombine(cells, model.split_model, extended_attributes)

    # interact with the environment
    cells = interact(cells, model, extended_attributes)

    # update the cells internally
    cells = react(
        cells,
        model,
        key,
        extended_attributes,
    )

    # apply forces to update cell positions
    forces = cells.motility_force + cells.mechanical_force
    cells = cells.replace(position=cells.position + model.delta_t * forces)

    return cells


def split_and_recombine(
    cells: Cell, split_model: SplitModel, extended_attributes=False
) -> Cell:
    # total number of cells (including inactive)
    num_cells = len(cells.parent)

    # update parent indices to current indices
    # (this invalidates the parent index temporarily)
    active = cells.active
    indices = jnp.arange(num_cells, dtype=jnp.int16)
    parent_indices = indices * active - (1 - active)
    cells = cells.replace(parent=parent_indices)

    # split all cells
    daughters_a, daughters_b = jax.vmap(split_cell, in_axes=(0, None, None))(
        cells,
        split_model,
        extended_attributes,
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
    # update cell state
    cells = model.react_model(cells, extended_attributes)

    # calculate motility forces
    cells = model.motility_model(cells, extended_attributes)

    # compute split probabilities and sample an action
    cells = model.split_prob_model(cells, key, extended_attributes)

    return cells


def split_cell(cell: Cell, split_model: SplitModel, extended_attributes=False):
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
    daughter_a_position = cell.position + division_plane * daughter_b_radius
    daughter_b_position = cell.position - division_plane * daughter_a_radius

    if extended_attributes:
        cell = cell.replace(
            volume_ratio=volume_ratio,
            division_plane=division_plane,
        )

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
