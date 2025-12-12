import jax
import jax.numpy as jnp

from .diffusion import DiffusionModel
from .mechanics import MechanicsModel
from .model import Model
from .motility import MotilityModel
from .react import ReactModel
from .secretion import SecretionModel
from .sensation import SensationModel
from .split import SplitModel
from .split_prob import SplitProbModel


def create_model(
    key,
    max_num_cells,
    cell_state_dims,
    num_molecules,
    delta_t,
    exploration_eps=0.0,
    morse_well_width=1.0,
    morse_well_depth=1.0,
    rl_discount_gamma=0.9,
    entropy_regularizer=0.0,
    direct_loss_weight=0.1,
):
    key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)

    react_model = ReactModel(
        cell_state_dims,
        cell_state_dims * 2,
        key=key1,
    )
    motility_model = MotilityModel(
        cell_state_dims,
        cell_state_dims * 2,
        3,
        key=key2,
    )
    split_prob_model = SplitProbModel(
        cell_state_dims,
        cell_state_dims * 2,
        eps=exploration_eps,
        key=key3,
    )

    split_model = SplitModel(
        cell_state_dims,
        3,
        cell_state_dims * 2,
        key=key4,
    )

    mechanics_model = MechanicsModel(morse_well_width, morse_well_depth)

    secretion_model = SecretionModel(
        num_molecules, cell_state_dims, cell_state_dims * 2, key5
    )
    diffusion_model = DiffusionModel(
        diffusion_coefs=jnp.ones(num_molecules),
        degradation_rates=jnp.ones(num_molecules),
    )
    sensation_model = SensationModel(
        cell_state_dims, num_molecules, hidden_dims=cell_state_dims * 2, key=key6
    )

    model = Model(
        max_num_cells=max_num_cells,
        cell_state_dims=cell_state_dims,
        num_molecules=num_molecules,
        delta_t=delta_t,
        react_model=react_model,
        motility_model=motility_model,
        split_prob_model=split_prob_model,
        split_model=split_model,
        mechanics_model=mechanics_model,
        secretion_model=secretion_model,
        diffusion_model=diffusion_model,
        sensation_model=sensation_model,
        rl_discount_gamma=rl_discount_gamma,
        entropy_regularizer=entropy_regularizer,
        direct_loss_weight=direct_loss_weight,
        key=key7,
    )
    return model
