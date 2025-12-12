import equinox as eqx

from ..cell import Cell
from ..mechanics import morse_force


class MechanicsModel(eqx.Module):
    morse_well_width: float
    morse_well_depth: float

    def __init__(self, morse_well_width, morse_well_depth):
        self.morse_well_width = float(morse_well_width)
        self.morse_well_depth = float(morse_well_depth)

    def __call__(self, cells: Cell, extended_attributes: bool = False):
        # mask out inactive cells
        force = morse_force(
            cells.position,
            cells.radius,
            cells.active,
            well_width=self.morse_well_width,
            well_depth=self.morse_well_depth,
        )
        return cells.replace(mechanical_force=force)
