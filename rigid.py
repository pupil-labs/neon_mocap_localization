import numpy as np


def fit_plane(centers, orient_towards=None):
    centroid = np.mean(centers, axis=1, keepdims=True)
    centered = centers - centroid

    # SVD on 3xN centered points
    try:
        U, _, _ = np.linalg.svd(centered)
    except Exception:
        # sometimes SVD does not converge on poor frames or frames
        # without apriltags, so just skip those few instances
        return None, None

    normal = U[:, 2]
    normal /= np.linalg.norm(normal)

    if orient_towards is not None:
        orient_towards = np.asarray(orient_towards, dtype=float)

        # if orient_towards.ndim == 1:
        # orient_towards = orient_towards[:, np.newaxis]

        ref_vec = orient_towards - centroid.squeeze()
        print(ref_vec)

        if np.dot(normal, ref_vec) > 0:
            normal = -normal

        U[:, 2] = normal

    U[:, 0] /= np.linalg.norm(U[:, 0])
    U[:, 1] /= np.linalg.norm(U[:, 1])
    U[:, 2] /= np.linalg.norm(U[:, 2])

    return centroid, U
