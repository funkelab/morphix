import equinox as eqx
import jax


class MotilityModel(eqx.Module):
    layers: tuple
    spatial_dims: int

    def __init__(self, cell_state_dims: int, hidden_dims: int, spatial_dims: int, key):
        key1, key2 = jax.random.split(key, 2)
        self.spatial_dims = spatial_dims
        self.layers = (
            eqx.nn.Linear(cell_state_dims, hidden_dims, key=key1),
            eqx.nn.LayerNorm(hidden_dims),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dims, spatial_dims, key=key2),
        )

    def __call__(self, cells, extended_attributes: bool = False):
        force = jax.vmap(self.motility_force)(cells.state)
        return cells.replace(motility_force=force)

    def motility_force(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        return x
