import matplotlib.pyplot as plt


def refocus_app(rgb_stack, index_map):
    """
    Refocusing app per spec:
      1) display an image in the focal stack
      2) ask a user to choose a scene point (with Matplotlib ginput)
      3) refocus to the image such that the scene point is focused
      4) terminate when the user chooses a point outside the image frame
    Args:
        rgb_stack: (H, W, 3, N) float array
        index_map: (H, W) int array with values in [0..N-1]
    """
    H, W, _, N = rgb_stack.shape
    current = N // 2

    fig, ax = plt.subplots()
    img = ax.imshow(rgb_stack[..., current])
    ax.set_title("Click to refocus | Click outside image to quit")
    ax.axis("off")

    def on_click(event):
        # Exit if the click is outside any axes (e.g., on the window background)
        if event.inaxes is None:
            plt.close(fig)
            return

        # Refocus using the per-pixel layer index
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if 0 <= x < W and 0 <= y < H:
            layer = int(index_map[y, x])
            layer = max(0, min(N - 1, layer))
            img.set_data(rgb_stack[..., layer])
            ax.set_title(f"Refocused to layer {layer} | Click outside image to quit")
            fig.canvas.draw_idle()
        else:
            # Exit if click landed outside image bounds
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
