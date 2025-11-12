import numpy as np
import matplotlib.pyplot as plt


def refocus_app(rgb_stack, index_map):
    """
    Minimal interactive refocusing viewer.
    - rgb_stack: (H, W, 3, N) float array
    - index_map: (H, W) integer array mapping pixel -> best layer (0..N-1)
    Click anywhere in the image to refocus to the layer that best focuses
    the clicked pixel, according to index_map. Click outside the image to quit.
    """
    H, W, _, N = rgb_stack.shape
    cur = N // 2

    fig, ax = plt.subplots()
    im = ax.imshow(rgb_stack[:, :, :, cur])
    ax.set_title(f"Refocus App — slice {cur+1}/{N}\nClick a point to refocus (click outside to exit)")
    plt.axis('off')
    plt.tight_layout()

    while True:
        pts = plt.ginput(1, timeout=-1)
        if not pts:
            break
        x, y = pts[0]  # (x, y) with origin at top-left
        if x < 0 or y < 0 or x >= W or y >= H:
            break

        j = int(round(x))
        i = int(round(y))
        j = max(0, min(W-1, j))
        i = max(0, min(H-1, i))

        target = int(index_map[i, j])
        target = max(0, min(N-1, target))

        if target != cur:
            cur = target
            im.set_data(rgb_stack[:, :, :, cur])
            ax.set_title(f"Refocus App — slice {cur+1}/{N}\nClick a point to refocus (click outside to exit)")
            plt.draw()

    plt.close(fig)
