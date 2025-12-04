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
    current_layer = N // 2

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rgb_stack[..., current_layer])
    ax.set_title(
        f"Layer {current_layer + 1}/{N} - Click to refocus | Click outside image to quit"
    )
    ax.axis("off")
    plt.tight_layout()

    print("Refocus App Started")
    print(f"Currently displaying layer {current_layer + 1}/{N}")
    print("Click on any point in the image to refocus.")
    print("Click outside the image to exit.")

    def on_click(event):
        nonlocal current_layer

        # Exit if the click is outside any axes (e.g., on the window background)
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            print("Exiting refocus app.")
            plt.close(fig)
            return

        x = event.xdata
        y = event.ydata
        col = int(round(x))
        row = int(round(y))

        # Exit if the click is outside the image bounds
        if row < 0 or row >= H or col < 0 or col >= W:
            print(f"Point ({col}, {row}) is outside the image. Exiting.")
            plt.close(fig)
            return

        # Get the best focused layer for this point from the index map
        target_layer = int(index_map[row, col])

        # Clamp to valid range, just in case
        if target_layer < 0:
            target_layer = 0
        elif target_layer >= N:
            target_layer = N - 1

        print(f"Clicked at pixel ({col}, {row})")
        print(f"Refocusing from layer {current_layer + 1} to layer {target_layer + 1}")

        # Animate the transition
        if target_layer != current_layer:
            step = 1 if target_layer > current_layer else -1

            for layer in range(current_layer, target_layer + step, step):
                if layer != current_layer:
                    img = rgb_stack[..., layer]
                    im.set_data(img)
                    ax.set_title(f"Layer {layer + 1}/{N} - Refocusing...")
                    plt.pause(0.05)

            current_layer = target_layer

        # Display the final refocused image
        refocused_img = rgb_stack[..., target_layer]
        im.set_data(refocused_img)
        ax.set_title(
            f"Layer {target_layer + 1}/{N} - Focused at ({col}, {row}) - "
            "Click to refocus again"
        )
        fig.canvas.draw_idle()
        print(f"Now displaying layer {target_layer + 1}/{N}")
        print("Click on another point to refocus, or click outside to exit.")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    print("Refocus app terminated.")
