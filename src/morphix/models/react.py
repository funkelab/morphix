import equinox as eqx
import jax

from ..cell import Cell


class ReactModel(eqx.Module):
    layers: tuple

    def __init__(self, cell_state_dims: int, hidden_dims: int, key):
        key1, key2 = jax.random.split(key, 2)
        self.layers = (
            eqx.nn.Linear(cell_state_dims, hidden_dims, key=key1),
            eqx.nn.LayerNorm(hidden_dims),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dims, cell_state_dims, key=key2),
        )

    def __call__(self, cells: Cell, extended_attributes: bool = False):
        state = jax.vmap(self.update_state)(cells.state)
        return cells.replace(state=state)

    def update_state(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        return cell_state + x
