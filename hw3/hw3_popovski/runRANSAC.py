import numpy as np
from computeHomography import compute_homography
from applyHomography import apply_homography


def run_ransac(Xs, Xd, ransac_n, eps):
    num_pts = Xs.shape[0]
    pts_id = np.arange(num_pts)
    inliers_id = np.array([])
    H = np.eye(3)  # H placeholder

    for iter in range(ransac_n):
        # ---------------------------
        # START ADDING YOUR CODE HERE
        # ---------------------------
        # Sample 4 unique correspondences
        # Sample 4 unique correspondences
        idx = np.random.choice(pts_id, size=4, replace=False)

        # Try to compute a trial homography (src->dest) on the minimal set
        try:
            H_try = compute_homography(Xs[idx], Xd[idx])
            if H_try is None:
                continue
        except Exception:
            continue  # degenerate sample, skip

        # Reproject ALL source points and compute Euclidean error in dest frame
        proj = apply_homography(H_try, Xs)  # (N,2)
        err = np.linalg.norm(proj - Xd, axis=1)

        # Inliers are points whose reprojection error is below eps
        inliers_mask = err < float(eps)
        n_inl = int(inliers_mask.sum())

        # Keep the best set (most inliers); refit H on all inliers if possible
        if n_inl > inliers_id.size:
            inliers_id = np.where(inliers_mask)[0]
            if n_inl >= 4:
                try:
                    H = compute_homography(Xs[inliers_mask], Xd[inliers_mask])
                except Exception:
                    H = H_try
            else:
                # Not enough to refit; keep the trial estimate
                H = H_try

        # ---------------------------
        # END ADDING YOUR CODE HERE
        # ---------------------------
        pass  # placeholder so for loop isn't empty.

    return inliers_id, H
