from abc import abstractmethod

import jax
import jax.numpy as jnp

from .cell import Cell


class Loss:
    """Base class for losses."""

    @abstractmethod
    def compute(self, trajectory: Cell) -> tuple[jax.Array, jax.Array]:
        """Main method to compute the loss.

        Args:
            trajectory:

                The simulated trajectory to compute the loss for.

        Returns:
            A loss tensor that should be shaped `(t, n)` , where `t` is the
            number of time steps and `n` the number of cells (including
            inactive cells).

            Some losses do not factorize over individual cells. In this case,
            this method can return a tensor shaped `(t, 1)`. The loss will then
            be broadcast later over all cells.
        """
        pass


class LineageLoss(Loss):
    """A loss to match a simulation to a target lineage.

    This loss computes the sum of squared differences of two mixture of
    Gaussians densities, evaluated at specific key points. The key points are
    the positions of the target cells, as well as additional points offset by
    `sigma` in each spatial direction (positive and negative). The `sigma`
    parameter also controls the width of the Gaussians.

    Args:
        target:

            The target lineage. Only `position` and `active` will be used.

        sigma:

            The variance of the Gaussians used in the mixture to fit positions.
    """

    def __init__(self, target, sigma=1.0):
        self.target = target
        self.sigma = sigma

    def compute(self, trajectory: Cell) -> tuple[jax.Array, jax.Array]:
        # map over time steps
        #
        # losses: (t,)
        losses = jax.vmap(self.timestep_loss)(trajectory, self.target)

        # ensure that losses are broadcastable over cells
        #
        # losses: (t, 1)
        return losses[:, None]

    def timestep_loss(self, cells, target):
        # compute keypoints around each cell and each target location
        #
        # key points are the center of the cell, plus one more key point in
        # each direction (positive and negative)
        #
        # key_points: (2k, d)   k = n + 2d
        offset = jnp.sqrt(self.sigma)
        key_points = jnp.concatenate(
            [
                target.position,
                target.position + jnp.array((offset, 0, 0)),
                target.position + jnp.array((0, offset, 0)),
                target.position + jnp.array((0, 0, offset)),
                target.position - jnp.array((offset, 0, 0)),
                target.position - jnp.array((0, offset, 0)),
                target.position - jnp.array((0, 0, offset)),
                cells.position,
                cells.position + jnp.array((offset, 0, 0)),
                cells.position + jnp.array((0, offset, 0)),
                cells.position + jnp.array((0, 0, offset)),
                cells.position - jnp.array((offset, 0, 0)),
                cells.position - jnp.array((0, offset, 0)),
                cells.position - jnp.array((0, 0, offset)),
            ]
        )

        def score(x):
            """Compute the MoG score for a given point `x`."""
            # x         : (d,)
            # key_points: (k, d)
            # dist      : (k,)
            dist = jnp.sum((key_points - x) ** 2 / self.sigma, axis=-1)
            # (k,)
            return jnp.exp(-dist)

        # compute key point scores for each target position
        # (n, k)
        target_scores = jax.vmap(score)(target.position)
        # ignore scores from inactive cells
        target_scores *= target.active[:, None]
        # sum over target positions
        # (k,)
        target_scores = target_scores.sum(axis=0)

        # compute key point scores for each predicted position
        # (n, k)
        trajectory_scores = jax.vmap(score)(cells.position)
        # ignore scores from inactive cells
        trajectory_scores *= cells.active[:, None]
        # sum over predicted positions
        # (k,)
        trajectory_scores = trajectory_scores.sum(axis=0)

        # compute squared error at key points
        # (k,)
        error = (target_scores - trajectory_scores) ** 2

        # average error over active target positions
        # timestep_loss: (,)
        timestep_loss = error.sum() / target.active.sum()

        return timestep_loss
