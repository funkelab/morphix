import json
import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
from rich import print as rprint

from .cell import Cell
from .models import create_model
from .simulation import simulation_step


def save_lineage(filename, lineage):
    """Save a whole lineage to file."""
    with open(filename, "wb") as file:
        pickle.dump(lineage, file)


def load_lineage(filename):
    """Read a lineage from file."""
    with open(filename, "rb") as file:
        return pickle.load(file)


def save_model(filename, model):
    """Save a model to file."""
    with open(filename, "wb") as file:
        file.write((json.dumps(model.hyperparameters()) + "\n").encode())
        eqx.tree_serialise_leaves(file, model)


def load_model(filename):
    """Read a model from file."""
    key = jax.random.key(0)
    with open(filename, "rb") as file:
        hyperparameters = json.loads(file.readline().decode())
        model_skeleton = create_model(
            key=key,
            **hyperparameters,
        )
        return eqx.tree_deserialise_leaves(file, model_skeleton)


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
    if values is None:
        return
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
    rprint(prefix, end="")
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


def print_cells(cells: Cell, i=None, limit=None):
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
            if i is not None:
                cell_id = f" {i}"
            else:
                cell_id = ""
            if cell.parent >= 0:
                print_color_values(
                    f"Cell{cell_id} at ",
                    cell.position,
                    min=min_pos,
                    max=max_pos,
                )
                print_color_values(
                    "\tradius     : ",
                    cell.radius,
                )
                print_color_values(
                    "\tmot. force : ",
                    cell.motility_force,
                    min=-1.0,
                    max=1.0,
                )
                print_color_values(
                    "\tmec. force : ",
                    cell.mechanical_force,
                    min=-1.0,
                    max=1.0,
                )
                print_color_values(
                    "\tvol ratio  : ",
                    cell.volume_ratio,
                    min=-1.0,
                    max=1.0,
                )
                print_color_values(
                    "\tdiv plane  : ",
                    cell.division_plane,
                    min=-1.0,
                    max=1.0,
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
            if not limit:
                limit = num_cells
            for i in range(limit):
                print_cells(jax.tree.map(lambda a: a[i], cells), i)
            if num_cells > limit:
                jax.debug.print("(and {} more)", num_cells - limit)


def print_simulation(model, num_timesteps, key, limit_cells=None):
    """Run a simulation and pretty-print each timestep."""
    subkey, key = jax.random.split(key, 2)
    cells = model.initialize_cells(subkey, extended_attributes=True)
    print()
    print()
    for t in range(num_timesteps):
        print(f"{t=}:")
        print_cells(cells, limit=limit_cells)
        subkey, key = jax.random.split(key, 2)
        cells = simulation_step(cells, model, subkey, extended_attributes=True)
