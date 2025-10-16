import pickle

import jax
import jax.numpy as jnp
from rich import print as rprint

from .cell import Cell
from .simulation import simulation_step


def save_lineage(filename, lineage):
    """Save a whole lineage to file."""
    with open(filename, "wb") as file:
        pickle.dump(lineage, file)


def load_lineage(filename):
    """Read a lineage from file."""
    with open(filename, "rb") as file:
        return pickle.load(file)


def print_color_values(
    prefix,
    values,
    min=0.0,
    max=1.0,
    max_elements=None,
    from_color=(100, 100, 100),
    to_color=(200, 200, 200),
):
    """Print (array) values with a color depending on the value."""
    is_array = True
    truncated = False
    if isinstance(values, jax.Array) and len(values.shape) > 0:
        values = [float(v.item()) for v in values]
    else:
        values = [values]
        is_array = False
    if max_elements and len(values) > max_elements:
        truncated = True
        values = values[:max_elements]
    if max - min < 1e-4:
        max = min + 1.0
    print(prefix, end="")
    if is_array:
        print("[", end="")

    def color(value):
        v = (value - min) / (max - min)
        r = int(v * to_color[0] + (1.0 - v) * from_color[0])
        g = int(v * to_color[1] + (1.0 - v) * from_color[1])
        b = int(v * to_color[2] + (1.0 - v) * from_color[2])
        return f"rgb({r},{g},{b})"

    rprint(
        " ".join(f"[{color(v)}]{v:.2f}[/]" for v in values),
        end="",
    )

    if is_array:
        if truncated:
            print(" ...", end="")
        print("]", end="")
    print()


def print_cells(cells: Cell):
    """Pretty print cells."""
    dims = len(cells.parent.shape)

    min_pos = cells.position.min()
    max_pos = cells.position.max()

    min_state = cells.state.min().item()
    max_state = cells.state.max().item()

    min_sec = cells.secretion.min().item()
    max_sec = cells.secretion.max().item()

    with jnp.printoptions(precision=2, floatmode="fixed", threshold=3, suppress=True):

        def print_callback(cell):
            if cell.parent >= 0:
                print_color_values(
                    "Cell at ",
                    cell.position,
                    min=min_pos,
                    max=max_pos,
                )
                print_color_values(
                    "\tradius     : ",
                    cell.radius,
                )
                print_color_values(
                    "\tstate      : ",
                    cell.state,
                    min=min_state,
                    max=max_state,
                    max_elements=5,
                    from_color=(69, 125, 255),
                    to_color=(23, 200, 41),
                )
                print_color_values(
                    "\tsecretion  : ",
                    cell.secretion,
                    min=min_sec,
                    max=max_sec,
                    max_elements=5,
                    from_color=(131, 23, 41),
                    to_color=(255, 69, 125),
                )
                print_color_values(
                    "\tp(split)   : ",
                    cell.p_split,
                    from_color=(255, 0, 0),
                    to_color=(0, 255, 0),
                )
                rprint(
                    "\tsplit?     :",
                    cell.split,
                )
                rprint(
                    "\tparent     :",
                    cell.parent,
                )
            else:
                rprint("<empty>")

        if dims == 0:
            jax.debug.callback(print_callback, cells)
        else:
            num_cells = len(cells.parent)
            for i in range(10):
                print_cells(jax.tree.map(lambda a: a[i], cells))
            if num_cells > 10:
                jax.debug.print("(and {} more)", num_cells - 10)


def print_simulation(model, num_timesteps, key):
    """Run a simulation and pretty-print each timestep."""
    subkey, key = jax.random.split(key, 2)
    cells = model.initialize_cells(subkey)
    print()
    print()
    for t in range(min(num_timesteps, 10)):
        print(f"{t=}:")
        print_cells(cells)
        subkey, key = jax.random.split(key, 2)
        cells = simulation_step(cells, model, subkey)
