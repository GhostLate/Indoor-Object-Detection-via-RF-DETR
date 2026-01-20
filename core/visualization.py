import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt


def draw_plots(depths_dict: dict[str, torch.Tensor | np.ndarray], show=True, save_path=None):
    """
    Helper to visualize multiple images/maps in a grid layout.
    """
    for name, depth in depths_dict.items():
        if isinstance(depth, torch.Tensor):
            depths_dict[name] = depth.detach().cpu().numpy()

    depth_len = len(depths_dict.keys())
    y = int(np.ceil(np.sqrt(depth_len)))
    x = int(np.ceil(depth_len / y))

    fig, axs = plt.subplots(x, y, layout='compressed', figsize=(20, 10))

    if x == 1 or y == 1:
        for i, (name, depth) in enumerate(depths_dict.items()):
            p = axs[i].imshow(depth)
            fig.colorbar(p, ax=axs[i])
            axs[i].set_title(name)
    else:
        for i, (name, depth) in enumerate(depths_dict.items()):
            xi = int(i // y)
            yi = int(i % y)
            p = axs[xi, yi].imshow(depth)
            fig.colorbar(p, ax=axs[xi, yi])
            axs[xi, yi].set_title(name)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()
