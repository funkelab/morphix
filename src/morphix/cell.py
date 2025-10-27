import equinox as eqx
import jax


class Cell(eqx.Module):
    log_p_motility: jax.Array
    position: jax.Array
    radius: jax.Array
    state: jax.Array
    secretion: jax.Array
    concentration: jax.Array
    parent: jax.Array  # -1 if not active, otherwise index into previous state
    p_split: jax.Array
    split: jax.Array
    motility_force: jax.Array
    mechanical_force: jax.Array

    # extended attributes
    volume_ratio: jax.Array | None = None
    division_plane: jax.Array | None = None

    @property
    def num_cells(self) -> int:
        if len(self.parent.shape) == 0:
            return 1
        else:
            return self.parent.shape[-1]

    @property
    def active(self):
        return self.parent >= 0

    def replace(
        self,
        **kwargs,
    ):
        return eqx.tree_at(
            lambda c: tuple(getattr(c, name) for name in kwargs.keys()),
            self,
            tuple(kwargs.values()),
        )
