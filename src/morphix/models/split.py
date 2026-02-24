import equinox as eqx
import jax
import jax.numpy as jnp

from .swiglu import SwiGLU


class SplitModel(eqx.Module):
    layers: tuple
    cell_state_dims: int
    spatial_dims: int
    max_volume_difference: float

    def __init__(
        self,
        cell_state_dims: int,
        spatial_dims: int,
        hidden_dims: int,
        key,
        max_volume_difference: float = 0.2,
    ):
        """Predict how to divide a cell.

        Given a cell state, output coefficients for how to split the state into
        two daughter cells, the ratio of their sizes, and the orientation of
        the division plane.

        Args:
            cell_state_dims:

                The size of the cell state.

            spatial_dims:

                The number of spatial dimensions (used to predict the division
                plane).

            hidden_dims:

                The size of the hidden layer of the MLP.

            key:

                RNG key to use for initialization.

            max_volume_difference:

                The maximum volume difference of the daughter cells, expressed
                in percentage of the parent volume. Controls the asymmetry of
                divisions by limiting the range of the predicted `volume_ratio`
                to `[0.5 - d/2, 0.5 + d/2]` (where `d` is
                `max_volume_difference`). A value of 0.0 will lead to symmetric
                divisions, a value of 1.0 does not restrict the size difference
                at all.
        """
        self.layers = (
            SwiGLU(
                cell_state_dims,
                hidden_dims,
                out_features=cell_state_dims + 1 + spatial_dims,
                key=key,
            ),
        )
        self.cell_state_dims = cell_state_dims
        self.spatial_dims = spatial_dims
        self.max_volume_difference = float(max_volume_difference)

    def __call__(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        state_ratio = jax.nn.sigmoid(x[: self.cell_state_dims])
        volume_ratio = (0.5 - self.max_volume_difference / 2) + jax.nn.sigmoid(
            x[self.cell_state_dims]
        ) * self.max_volume_difference
        division_plane = x[self.cell_state_dims + 1 :]
        division_plane = division_plane / jnp.clip(
            jnp.linalg.norm(division_plane), min=1e-10
        )
        return state_ratio, volume_ratio, division_plane
