import equinox as eqx
import jax

from morphix.models.swiglu import SwiGLU

from ..cell import Cell


class ReactModel(eqx.Module):
    layers: tuple

    def __init__(self, cell_state_dims: int, hidden_dims: int, key):
        self.layers = (SwiGLU(cell_state_dims, hidden_dims, key=key),)

    def __call__(self, cells: Cell, extended_attributes: bool = False):
        state = jax.vmap(self.update_state)(cells.state)
        return cells.replace(state=state)

    def update_state(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        return cell_state + x
