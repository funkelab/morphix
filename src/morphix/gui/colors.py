import numpy as np
import umap


def states_to_rgb(states: np.ndarray, mode: str = "umap") -> np.ndarray:
    """Use UMAP to embed the state in three dimensions for RGB."""
    states = states.reshape(-1, states.shape[-1])
    if mode == "umap":
        colors = umap.UMAP(
            n_components=3,
            random_state=1912,
        ).fit_transform(states)
    elif mode == "direct":
        colors = states[:, :3]
    else:
        raise RuntimeError(f"Unknown mode '{mode}'")

    assert isinstance(colors, np.ndarray)
    colors_min = colors.min(axis=0)
    colors_max = colors.max(axis=0)
    return (colors - colors_min) / (colors_max - colors_min)
