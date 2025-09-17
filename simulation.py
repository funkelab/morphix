import jax.numpy as jnp
import jax
from cell import Cell
from models import ReactModel, SplitModel
from indexing import masks_to_indices
from functools import partial


def diffuse(cells: Cell) -> Cell:
    # do nothing for now
    return cells


def react(cells: Cell, react_model: ReactModel) -> Cell:
    return cells.replace(state=react_model(cells.state))


def split_cell(cell: Cell, index):
    # do nothing for now, just copy the cell
    daughter_a = cell
    daughter_b = cell

    active = cell.parent >= 0
    parent_index = index * active + -1 * (1 - active)
    daughter_a = daughter_a.replace(parent=parent_index)
    daughter_b = daughter_b.replace(parent=parent_index)

    return (daughter_a, daughter_b)


def split(cells: Cell, split_model: SplitModel) -> Cell:
    # get active cells
    active = cells.parent >= 0

    # total number of cells (including inactive)
    num_cells = len(cells.parent)

    # ask split model for split probabilities
    split_probs = split_model(cells.state)

    # sample an action (split or not)
    do_split = split_model.sample(split_probs)

    # record split probabilities and action
    parents = cells.replace(p_split=split_probs, split=do_split)

    # split all parents (regardless of do_split)
    daughters_a, daughters_b = jax.vmap(split_cell)(parents, jnp.arange(num_cells))

    keep_parents = (~do_split) & active
    keep_daughters = do_split & active

    # jax.debug.print("split_probs: {split_probs}", split_probs=split_probs)
    # jax.debug.print("do_split: {do_split}", do_split=do_split)

    return parents, daughters_a, daughters_b, keep_parents, keep_daughters


@partial(jax.jit, static_argnames=("max_num_cells",))
def recombine(
    parents: Cell,
    daughters_a: Cell,
    daughters_b: Cell,
    keep_parents: jax.Array,
    keep_daughters: jax.Array,
    max_num_cells: int,
) -> Cell:
    indices = masks_to_indices(keep_parents, keep_daughters)
    # recombine and trim
    cells = jax.tree.map(
        lambda p, a, b: jnp.concatenate((p, a, b))[indices][:max_num_cells],
        parents,
        daughters_a,
        daughters_b,
    )

    return cells


@partial(jax.jit, static_argnames=("max_num_cells",))
def simulation_step(
    cells: Cell, react_model: ReactModel, split_model: SplitModel, max_num_cells: int
) -> Cell:
    cells = diffuse(cells)
    cells = react(cells, react_model)
    parents, daughters_a, daughters_b, keep_parents, keep_daughters = split(
        cells, split_model
    )
    cells = recombine(
        parents,
        daughters_a,
        daughters_b,
        keep_parents,
        keep_daughters,
        max_num_cells,
    )
    return cells


@partial(jax.jit, static_argnames=("num_iterations",))
def simulate(
    cells: Cell, react_model: ReactModel, split_model: SplitModel, num_iterations: int
) -> Cell:
    num_cells = len(cells.parent)

    def step(cells, _):
        cells = simulation_step(cells, react_model, split_model, num_cells)
        return cells, cells

    _, all_cells = jax.lax.scan(step, cells, length=num_iterations)
    return all_cells
