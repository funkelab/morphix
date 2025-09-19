import jax
import jax.numpy as jnp
from cell import Cell


def straight_through(x, f):
    return jax.lax.stop_gradient(f(x)) + (x - jax.lax.stop_gradient(x))


def step_function(x, threshold):
    return straight_through(x, lambda x: x >= threshold)


def print_cells(cells: Cell):
    dims = len(cells.parent.shape)

    with jnp.printoptions(precision=3, threshold=5, suppress=True):

        def print_callback(cell):
            if cell.parent >= 0:
                print(
                    f"Cell {cell.state}, p_split {cell.p_split:.3f}, split {cell.split}, parent {cell.parent}"
                )
            else:
                print("<empty>")

        if dims == 0:
            jax.debug.callback(print_callback, cells)
        else:
            num_cells = len(cells.parent)
            jax.vmap(print_cells)(jax.tree.map(lambda a: a[:10], cells))
            if num_cells > 10:
                jax.debug.print("(and {} more)", num_cells - 10)
