import colorsys

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


def random_colors(num_colors: int) -> np.ndarray:
    """Create random colors that are maximally distinguishable."""
    if num_colors <= 0:
        return np.empty((0, 3))

    hues = (np.linspace(0, 1, num_colors, endpoint=False) + np.random.rand()) % 1.0
    rgb_colors = np.array(
        [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues], dtype=np.float32
    )

    return rgb_colors
