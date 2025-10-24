import numpy as np
from scipy.ndimage import distance_transform_edt as bwdist


def blend_image_pair(wrapped_imgs, masks, wrapped_imgd, maskd, mode):
    Hs, Ws, Cs = wrapped_imgs.shape
    Hd, Wd, Cd = wrapped_imgd.shape

    assert (Hs == Hd) and (Ws == Wd) and (Cs == Cd)

    assert wrapped_imgs.dtype == wrapped_imgd.dtype
    assert wrapped_imgs.dtype in [np.uint8, np.float32, np.float64]

    out_img = np.zeros((Hs, Ws, Cs), dtype=np.float64)
    input_type = wrapped_imgs.dtype

    # convert to float64 to avoid overflow/underflow when multiplying with
    # the weighted mask
    wrapped_imgs = wrapped_imgs.astype(np.float64)
    wrapped_imgd = wrapped_imgd.astype(np.float64)

    binary_mask_s = masks > 0
    binary_mask_d = maskd > 0

    for c in range(Cs):
        channel_out = np.zeros((Hs, Ws), dtype=np.float64)
        S = wrapped_imgs[:, :, c]
        D = wrapped_imgd[:, :, c]
        if mode == "overlay":
            # s first, then d overwrites s wherever there is overlap.
            channel_out[binary_mask_s] = S[binary_mask_s]
            channel_out[binary_mask_d] = D[binary_mask_d]
        elif mode == "blend":
            # ---------------------------
            # ADD YOUR CODE HERE
            # ---------------------------
            for c in range(Cs):
                src = wrapped_imgs[..., c].astype(np.float64)
                dst = wrapped_imgd[..., c].astype(np.float64)

                if mode == "overlay":
                    channel_out = src.copy()
                    channel_out[maskd.astype(bool)] = dst[maskd.astype(bool)]

                elif mode == "blend":
                    m1 = (masks > 0).astype(np.uint8)
                    m2 = (maskd > 0).astype(np.uint8)

                    union = (m1 | m2).astype(np.uint8)
                    if not np.any(union):
                        channel_out = np.zeros_like(src, dtype=np.float64)
                    else:
                        # Distance to boundary *inside* each region
                        d1 = bwdist(m1) * m1
                        d2 = bwdist(m2) * m2

                        wsum = d1 + d2 + 1e-8
                        w1 = np.where(union, d1 / wsum, 0.0)
                        w2 = np.where(union, d2 / wsum, 0.0)

                        # Where only one is present, give it full weight
                        w1[(m1 == 1) & (m2 == 0)] = 1.0
                        w2[(m2 == 1) & (m1 == 0)] = 1.0

                        channel_out = w1 * src + w2 * dst
            else:
                raise ValueError(f"Unknown blending mode: {mode}")

        out_img[:, :, c] = channel_out

    # convert out_img to right type
    if input_type == np.uint8:
        out_img = np.clip(np.round(out_img), 0, 255).astype(np.uint8)
    elif input_type == np.float32:
        out_img = out_img.astype(np.float32)

    return out_img

