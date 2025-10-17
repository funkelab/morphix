import jax
import jax.numpy as jnp


def morse_force(positions, radii, active, well_width, well_depth):
    def potential_function(p):
        # we are only interested in the upper triangular pairs
        trui_indices = jnp.triu_indices(p.shape[0], k=1)

        # compute pairwise distances (eps before sqrt for finite gradients)
        distances = jnp.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1) + 1e-8)
        distances = distances[trui_indices]

        # compute pairwise equilibrium distances
        equilibrium_distances = radii[:, None] + radii[None, :]
        equilibrium_distances = equilibrium_distances[trui_indices]

        # find pairs of active cells
        active_pairs = active[:, None] * active[None, :]
        active_pairs = active_pairs[trui_indices]

        # compute morse potential
        potentials = (
            well_depth
            * (1 - jnp.exp(-1.0 / well_width * (distances - equilibrium_distances)))
            ** 2
        )

        # zero-out potentials from inactive cells
        potentials = potentials * active_pairs

        # discard duplicates and diagonal entries
        return potentials.sum()

    # compute gradient of morse potential wrt to positions
    grad = jax.grad(potential_function)(positions)

    # forces are negative gradients of potential function
    forces = -grad

    return forces
