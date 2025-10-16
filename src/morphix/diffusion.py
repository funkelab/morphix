import jax.numpy as jnp


def steady_state_concentrations(
    positions,
    radii,
    secretion_rates,
    diffusion_coefs,
    degradation_rates,
    active,
):
    # compute pairwise distances (eps before sqrt for finite gradients) and
    # clip them to be at least the radius of a cell to get consistent
    # concentrations for molecules secreted by the sensing cell
    distances = jnp.clip(
        jnp.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1) + 1e-8),
        min=radii,
    )

    # diffusion_coefs: (num_molecules,)
    # secretion_rates: (num_cells, num_molecules)
    # distances      : (num_cells, num_cells)

    # prepare arrays for broadcasting
    distances = distances[None, :, :]
    lambdas = jnp.sqrt(degradation_rates / diffusion_coefs)[:, None, None]
    secretion_rates = secretion_rates.T[:, None, :]
    diffusion_coefs = diffusion_coefs[:, None, None]
    # distances      : (1, num_cells, num_cells)
    # lambdas        : (num_molecules, 1, 1)
    # secretion_rates: (num_molecules, 1, num_cells)
    # diffusion_coefs: (num_molecules, 1, 1)

    # compute the concentrations using Green's function
    return jnp.sum(
        # (num_molecules, 1, num_cells)
        secretion_rates
        /
        # (num_molecules, num_cells, num_cells)
        (4 * jnp.pi * diffusion_coefs * distances)
        *
        # (num_molecules, num_cells, num_cells)
        jnp.exp(-lambdas * distances),
        axis=-1,
    ).T
