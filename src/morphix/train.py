from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .cell import Cell
from .simulation import simulate


@partial(
    jax.jit,
    static_argnames=(
        "static",
        "loss_function",
        "num_timesteps",
        "optimizer",
        "batch_size",
        "debug",
    ),
)
def train_step(
    params: optax.Params,
    static: eqx.Module,
    loss_function: Callable[[Cell, float], tuple[jax.Array, jax.Array]],
    num_timesteps: int,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: jax.Array,
    batch_size: int = 1,
    debug: bool = False,
):
    """Perform a single training step.

    Args:
        params:

            The trainable parameters of the model.

        static:

            All other static variables of the model.

        loss_function:

            A custom loss function to compute the lineage and properties losses
            for a given timestep.

            The loss function takes `(cells, t)` as input. `cells` is the set
            of all cells at time `t`. `cells` includes inactive cells. `t` is
            the simulation time (`timestep * model.delta_t`).

            It has to return two loss values: `(lineage_loss, cell_losses)`.
            `lineage_loss` is a scalar and `cell_losses` should have shape
            `(n,)`, where `n` is the number of cells (including inactive
            cells). Losses for inactive cells can be arbitrary and will be
            ignored (zeroed-out later on).

            The lineage loss scores the topology of the lineage and will be
            used in a reinforcement learning method to adjust the split
            probabilities of the model. This loss does not have to be
            differentiable.

            The cell losses are differentiable losses on arbitrary attributes
            of the cells (e.g., their position, size, or state). Those losses
            will be directly optimized.

        num_timesteps:

            The number of timesteps to simulate.

        optimizer:

            The optax optimizer and state to use.

        opt_state:

            The optax optimizer and state to use.

        key:

            JAX random key.

        batch_size:

            The number of simulation runs to average before updating the
            gradient. Increasing the batch size can help decrease the variance
            of the reinforcement learning loss.

        debug:

            If set, run in debug mode and print extra information.

    """

    @partial(jax.jit, static_argnames=("static",))
    @partial(jax.value_and_grad, has_aux=True)
    def batch_loss_grad(params, static, key):
        keys = jax.random.split(key, batch_size)
        model = eqx.combine(params, static)
        loss, details = jax.vmap(
            lambda k: simulation_loss(model, loss_function, num_timesteps, k, debug)
        )(keys)
        return loss.mean(), details

    (loss, details), grad = batch_loss_grad(params, static, key)

    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params, opt_state, details


def simulation_loss(
    model,
    loss_function,
    num_timesteps: int,
    key,
    debug: bool = False,
) -> tuple[jax.Array, dict]:
    """Run the simulation and compute the loss for the given loss function."""
    # run the simulation
    trajectory = simulate(model, num_timesteps, key, debug)

    # compute the loss
    loss, details = trajectory_loss(
        trajectory,
        loss_function,
        num_timesteps,
        model.delta_t,
        debug,
    )

    details["trajectory"] = trajectory
    return loss, details


def trajectory_loss(
    cells: Cell,
    loss_function,
    num_timesteps,
    delta_t,
    debug: bool = False,
) -> tuple[jax.Array, dict]:
    """Compute the loss for an entire trajectory."""
    # map loss function over time
    lineage_loss, cell_losses = jax.vmap(loss_function)(
        cells, jnp.arange(num_timesteps) * delta_t
    )

    # turn (non-differentiable) lineage losses into reinforcement learning
    # losses
    rl_losses = reinforcement_losses(cells, lineage_loss)

    # combine losses
    losses = rl_losses + cell_losses

    # zero-out losses of inactive cells
    active = cells.active
    losses *= active

    if debug:
        jax.debug.print("lineage loss: {}", lineage_loss)
        jax.debug.print("cell losses: {}", cell_losses)
        jax.debug.print("active: {}", active)
        jax.debug.print("losses: {}", losses)

    # compute mean loss over all active cells
    loss = losses.sum() / active.sum()

    details = {
        "lineage_loss": lineage_loss,
        "cell_losses": cell_losses,
        "losses": losses,
    }

    return loss, details


def reinforcement_losses(cells, lineage_loss):
    """Compute reinforcement losses given (non-differentiable) lineage losses."""
    # compute log probabilities of each action taken
    #
    # log_p_action: (t, n)
    log_p_action = jnp.log(
        cells.p_split * cells.split + (1.0 - cells.p_split) * (1 - cells.split)
    )

    # compute "losses-to-go" for each timestep, i.e., sum of future losses
    #
    # losses_to_go: (t,)
    losses_to_go = jnp.cumsum(lineage_loss[::-1], axis=0)[::-1]

    # compute per-cell reinforcement losses by broadcasting losses-to-go over
    # all cells (this includes inactive cells, which will be zeroed-out later)
    #
    # losses: (t, n)
    losses = log_p_action * losses_to_go[:, None]

    return losses
