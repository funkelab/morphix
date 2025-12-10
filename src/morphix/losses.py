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

        cell_count_penalty:

            The penalty to add to the loss for each unaccounted cell (in either
            target or simulation). Should be higher than the loss for a cell
            that is in the wrong location (max value ~3.2) such that a wrong
            number of active cells is penalized more than the correct number in
            the wrong locations.
    """

    def __init__(self, target, sigma=1.0, cell_count_penalty=10.0):
        self.target = target
        self.num_timesteps = target.position.shape[0]
        # indices to create "prev" targets and trajectories
        self.prev_indices = jnp.array([0, *range(0, self.num_timesteps - 1)])
        self.prev_target = target.replace(
            position=target.position[self.prev_indices],
            parent=target.parent[self.prev_indices],
        )
        self.sigma = sigma
        self.cell_count_penalty = cell_count_penalty

    def compute(self, trajectory: Cell) -> tuple[jax.Array, jax.Array]:
        # map over pairs of time steps

        # create "prev" cells by repeating first time step and dropping last
        prev_trajectory = jax.tree.map(lambda a: a[self.prev_indices], trajectory)
        # losses: (t-1,)
        losses = jax.vmap(self.timestep_loss)(
            prev_trajectory,
            trajectory,
            self.prev_target,
            self.target,
        )

        # ensure that losses are broadcastable over cells
        #
        # losses: (t, 1)
        return losses[:, None]

    def timestep_loss(self, prev_cells, cells, prev_target, target):
        # get positions of cells and targets in previous time step
        prev_cell_position = prev_cells.position[cells.parent]
        prev_target_position = prev_target.position[target.parent]

        # create movement vectors (from -> to, concatenated) for cells and targets
        # cell_movements: (n, 2d)
        # target_movements: (n, 2d)
        cell_movements = jnp.concatenate(
            (cells.position, prev_cell_position),
            axis=-1,
        )
        target_movements = jnp.concatenate(
            (target.position, prev_target_position),
            axis=-1,
        )

        # compute keypoints around each cell and each target movement
        #
        # key points are the movement, plus one more key point in each
        # direction (positive and negative)
        #
        # key_points: (2k, 2d)   k = n + 4d
        offset = jnp.sqrt(self.sigma)
        key_points = jnp.concatenate(
            [
                target_movements,
                target_movements + jnp.array((offset, 0, 0, 0, 0, 0)),
                target_movements + jnp.array((0, offset, 0, 0, 0, 0)),
                target_movements + jnp.array((0, 0, offset, 0, 0, 0)),
                target_movements + jnp.array((0, 0, 0, offset, 0, 0)),
                target_movements + jnp.array((0, 0, 0, 0, offset, 0)),
                target_movements + jnp.array((0, 0, 0, 0, 0, offset)),
                target_movements - jnp.array((offset, 0, 0, 0, 0, 0)),
                target_movements - jnp.array((0, offset, 0, 0, 0, 0)),
                target_movements - jnp.array((0, 0, offset, 0, 0, 0)),
                target_movements - jnp.array((0, 0, 0, offset, 0, 0)),
                target_movements - jnp.array((0, 0, 0, 0, offset, 0)),
                target_movements - jnp.array((0, 0, 0, 0, 0, offset)),
                cell_movements,
                cell_movements + jnp.array((offset, 0, 0, 0, 0, 0)),
                cell_movements + jnp.array((0, offset, 0, 0, 0, 0)),
                cell_movements + jnp.array((0, 0, offset, 0, 0, 0)),
                cell_movements + jnp.array((0, 0, 0, offset, 0, 0)),
                cell_movements + jnp.array((0, 0, 0, 0, offset, 0)),
                cell_movements + jnp.array((0, 0, 0, 0, 0, offset)),
                cell_movements - jnp.array((offset, 0, 0, 0, 0, 0)),
                cell_movements - jnp.array((0, offset, 0, 0, 0, 0)),
                cell_movements - jnp.array((0, 0, offset, 0, 0, 0)),
                cell_movements - jnp.array((0, 0, 0, offset, 0, 0)),
                cell_movements - jnp.array((0, 0, 0, 0, offset, 0)),
                cell_movements - jnp.array((0, 0, 0, 0, 0, offset)),
            ]
        )

        def score(x):
            """Compute the MoG score for a given point `x`."""
            # x         : (2d,)
            # key_points: (k, 2d)
            # dist      : (k,)
            dist = jnp.sum((key_points - x) ** 2 / self.sigma, axis=-1)
            # (k,)
            return jnp.exp(-dist)

        # compute key point scores for each target movement
        # (n, k)
        target_scores = jax.vmap(score)(target_movements)
        # ignore scores from inactive cells
        target_scores *= target.active[:, None]
        # sum over target positions
        # (k,)
        target_scores = target_scores.sum(axis=0)

        # compute key point scores for each predicted movement
        # (n, k)
        trajectory_scores = jax.vmap(score)(cell_movements)
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

        # add a cell count difference penalty
        timestep_loss += (
            jnp.abs(target.active.sum() - cells.active.sum()) * self.cell_count_penalty
        )

        return timestep_loss
