import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def refocus_app(rgb_stack, index_map):
    """
    Interactive refocus with tuned thresholds:
      • click strawberries/bottle/below → foreground focus
      • click stairs → all-focus average
      • click people → background
    """

    num_layers = rgb_stack.shape[3]
    h, w = index_map.shape

    # --- Normalize and smooth index map ---
    smoothed = gaussian_filter(index_map.astype(float), sigma=2)
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

    fig, ax = plt.subplots()
    current = rgb_stack[..., num_layers // 2]
    img = ax.imshow(current)
    plt.title("Click an area to refocus (outside image to quit)")

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            plt.close()
            return

        x, y = int(event.xdata), int(event.ydata)
        y = np.clip(y, 0, h - 1)
        x = np.clip(x, 0, w - 1)
        d = smoothed[y, x]

        # --- Heuristic thresholds tuned to your stack ---
        # Lower half of image is all considered foreground
        if y > h * 0.55 or d < 0.55:
            target = 2  # near front (strawberries + bottle)
            img.set_data(rgb_stack[..., target])
            plt.title("→ Focus: Foreground (bottle + strawberries)")

        elif d < 0.8:
            all_focus = np.mean(rgb_stack, axis=3).astype(rgb_stack.dtype)
            img.set_data(all_focus)
            plt.title("→ Focus: All-focus (stairs)")

        else:
            target = num_layers - 2
            img.set_data(rgb_stack[..., target])
            plt.title("→ Focus: Background (people)")

        plt.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
