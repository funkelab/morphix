import equinox as eqx
import jax
import jax.numpy as jnp

from ..cell import Cell
from .diffusion import DiffusionModel
from .mechanics import MechanicsModel
from .motility import MotilityModel
from .react import ReactModel
from .secretion import SecretionModel
from .sensation import SensationModel
from .split import SplitModel
from .split_prob import SplitProbModel


class Model(eqx.Module):
    # simulation parameters
    max_num_cells: int
    cell_state_dims: int
    num_molecules: int
    delta_t: float
    num_initial_cells: int
    initial_cell_states: jax.Array
    initial_cell_positions: jax.Array
    # models
    react_model: ReactModel
    motility_model: MotilityModel
    split_prob_model: SplitProbModel
    split_model: SplitModel
    mechanics_model: MechanicsModel
    secretion_model: SecretionModel
    diffusion_model: DiffusionModel
    sensation_model: SensationModel
    # learning parameters
    rl_discount_gamma: float
    entropy_regularizer: float
    direct_loss_weight: float

    def __init__(
        self,
        max_num_cells,
        cell_state_dims,
        num_molecules,
        delta_t,
        react_model,
        motility_model,
        split_prob_model,
        split_model,
        mechanics_model,
        secretion_model,
        diffusion_model,
        sensation_model,
        rl_discount_gamma,
        entropy_regularizer,
        direct_loss_weight,
        key,
    ):
        # hyperparameters

        self.max_num_cells = max_num_cells
        self.cell_state_dims = cell_state_dims
        self.num_molecules = num_molecules
        self.delta_t = float(delta_t)
        self.rl_discount_gamma = float(rl_discount_gamma)
        self.entropy_regularizer = float(entropy_regularizer)
        self.direct_loss_weight = float(direct_loss_weight)

        # sub-models

        self.react_model = react_model
        self.motility_model = motility_model
        self.split_prob_model = split_prob_model
        self.split_model = split_model
        self.mechanics_model = mechanics_model
        self.secretion_model = secretion_model
        self.diffusion_model = diffusion_model
        self.sensation_model = sensation_model

        # initial state

        self.num_initial_cells = (
            1 if initial_cell_positions is None else initial_cell_positions.shape[0]
        )
        self.initial_cell_states = jax.random.uniform(
            key=key, shape=(max_num_cells, cell_state_dims)
        )
        self.initial_cell_positions = jnp.zeros(
            (self.max_num_cells, self.motility_model.spatial_dims),
            dtype=jnp.float32,
        )
        if initial_cell_positions is not None:
            self.initial_cell_positions = self.initial_cell_positions.at[
                : self.num_initial_cells
            ].set(initial_cell_positions)

    def initialize_cells(self, key, extended_attributes=False):
        empty = jnp.zeros((), dtype=jnp.float32)

        if extended_attributes:
            extended_attrs = {
                "volume_ratio": jnp.zeros((self.max_num_cells,), dtype=jnp.float32),
                "division_plane": jnp.zeros(
                    (self.max_num_cells, self.motility_model.spatial_dims),
                    dtype=jnp.float32,
                ),
            }
        else:
            extended_attrs = {}

        cells = Cell(
            position=self.initial_cell_positions,
            radius=jnp.ones((self.max_num_cells,), dtype=jnp.float32),
            state=self.initial_cell_states,
            secretion=jnp.zeros(
                (self.max_num_cells, self.num_molecules), dtype=jnp.float32
            ),
            concentration=jnp.zeros(
                (self.max_num_cells, self.num_molecules), dtype=jnp.float32
            ),
            parent=(
                (-jnp.ones((self.max_num_cells,), dtype=jnp.int16))
                .at[: self.num_initial_cells]
                .set(0)
            ),
            # filled in below
            p_split=empty,
            # filled in below
            split=empty,
            motility_force=jnp.zeros(
                (self.max_num_cells, self.motility_model.spatial_dims),
                dtype=jnp.float32,
            ),
            mechanical_force=jnp.zeros(
                (self.max_num_cells, self.motility_model.spatial_dims),
                dtype=jnp.float32,
            ),
            **extended_attrs,
        )

        # initial positions
        cells = self.motility_model(cells, extended_attributes)

        # initial split decisions
        cells = self.split_prob_model(cells, key, extended_attributes)

        return cells

    def partition(self):
        """Partition the model into (parameters, static)."""
        return eqx.partition(self, eqx.is_inexact_array)

    def hyperparameters(self):
        """Return a dictionary of hyperparameters needed to create a similar model."""
        return {
            "max_num_cells": self.max_num_cells,
            "cell_state_dims": self.cell_state_dims,
            "num_molecules": self.num_molecules,
            "delta_t": self.delta_t,
            "rl_discount_gamma": self.rl_discount_gamma,
            "entropy_regularizer": self.entropy_regularizer,
            "direct_loss_weight": self.direct_loss_weight,
        }
