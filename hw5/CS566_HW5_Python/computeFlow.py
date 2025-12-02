import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import match_template


def compute_flow(img1, img2, search_radius, template_radius, grid_MN):
    # Check images have the same dimensions, and resize if necessary
    if img2.shape != img1.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Get number of rows and cols for output grid
    M = grid_MN[0]
    N = grid_MN[1]

    H, W = img1.shape[:2]
    # locations where we estimate the flow
    grid_y = np.round(np.linspace(template_radius + 1, H - template_radius, M)).astype(
        int
    )
    grid_x = np.round(np.linspace(template_radius + 1, W - template_radius, N)).astype(
        int
    )

    # allocate matrices where we will store the computed optical flow
    U = np.zeros((M, N))  # horizontal motion
    V = np.zeros((M, N))  # vertical motion

    # compute flow for each grid patch
    for i in range(M):
        for j in range(N):
            # ------------- PLEASE FILL IN THE NECESSARY CODE WITHIN THE FOR LOOP -----------------
            # Note: Wherever there are questions mark you should write
            # code and fill in the correct values there. You will need
            # to write more lines of code to obtain the correct values to
            # input in the questions marks.
            # extract the current patch/window (template)
            col = grid_x[j]
            row = grid_y[i]

            # Clamp to valid range
            col = np.clip(col, 0, W - 1)
            row = np.clip(row, 0, H - 1)

            # template window in img1 around (row, col)
            src_y_start = max(0, row - template_radius)
            src_y_end = min(H - 1, row + template_radius)
            src_x_start = max(0, col - template_radius)
            src_x_end = min(W - 1, col + template_radius)

            template = img1[src_y_start : src_y_end + 1, src_x_start : src_x_end + 1]

            # search window in img2 around (row, col)
            dest_y_start = max(0, row - search_radius)
            dest_y_end = min(H - 1, row + search_radius)
            dest_x_start = max(0, col - search_radius)
            dest_x_end = min(W - 1, col + search_radius)

            search_area = img2[
                dest_y_start : dest_y_end + 1, dest_x_start : dest_x_end + 1
            ]

            # If search area or template is degenerate, skip
            if template.size == 0 or search_area.size == 0:
                continue

            # If search area is smaller than template (can happen at borders), skip
            if (
                search_area.shape[0] < template.shape[0]
                or search_area.shape[1] < template.shape[1]
            ):
                U[i, j] = 0.0
                V[i, j] = 0.0
                continue

            # compute correlation
            corr_map = match_template(search_area, template)

            # Look at the correlation map and find the best match
            # The best match will have the Maximum Correlation value
            max_ind = np.argmax(corr_map)
            # Convert the index into row and col
            max_ind_row, max_ind_col = np.unravel_index(max_ind, corr_map.shape)

            # top-left of best match in img2
            best_y0 = dest_y_start + max_ind_row
            best_x0 = dest_x_start + max_ind_col

            # center of template
            th, tw = template.shape[:2]
            best_row = best_y0 + th // 2
            best_col = best_x0 + tw // 2

            # express peak location as offset from template location
            U[i, j] = best_col - col
            V[i, j] = row - best_row

    # Any post-processing or denoising needed on the flow

    # plot the flow vectors
    fig, ax = plt.subplots()
    ax.imshow(img1, cmap='gray')
    ax.quiver(grid_x, grid_y, U, V, 2, color='y', linewidth=1.3)
    fig.canvas.draw()

    # Convert the figure directly into an image matrix
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[..., 0:3]
    plt.close(fig)

    return img
