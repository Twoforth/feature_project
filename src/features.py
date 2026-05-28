import cv2
import numpy as np

from preprocess import (
    norm_vec,
    filt3,
    read_image,
    sobel_grad,
    grid_mean,
)


def edge_feature(gray):
    _, _, mag, _ = sobel_grad(gray)

    th = mag.mean() + 0.5 * mag.std()
    edge = (mag > th).astype(np.float32)

    grid = grid_mean(edge, 8, 8)

    stats = np.array(
        [
            edge.mean(),
            mag.mean(),
            mag.std(),
            mag.max(),
        ],
        dtype=np.float32,
    )

    return norm_vec(np.concatenate([grid, stats]))


def harris_feature(gray):
    gx, gy, _, _ = sobel_grad(gray)

    ixx = gx * gx
    iyy = gy * gy
    ixy = gx * gy

    box = np.ones((3, 3), dtype=np.float32) / 9.0

    sxx = filt3(ixx, box)
    syy = filt3(iyy, box)
    sxy = filt3(ixy, box)

    k = 0.04
    det = sxx * syy - sxy ** 2
    trace = sxx + syy
    r = det - k * trace ** 2

    if r.max() > 0:
        th = 0.01 * r.max()
        corners = r > th
    else:
        corners = np.zeros_like(r, dtype=bool)

    local_max = np.ones_like(corners, dtype=bool)

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue

            shifted = np.roll(np.roll(r, dy, axis=0), dx, axis=1)
            local_max &= r >= shifted

    corners = corners & local_max

    corners[0, :] = False
    corners[-1, :] = False
    corners[:, 0] = False
    corners[:, -1] = False

    corner_map = corners.astype(np.float32)
    grid = grid_mean(corner_map, 4, 4)

    stats = np.array(
        [
            corner_map.mean(),
            max(r.max(), 0),
            r.mean(),
            r.std(),
        ],
        dtype=np.float32,
    )

    return norm_vec(np.concatenate([grid, stats]))


def hog_feature(gray, cell=16, bins=9):
    _, _, mag, ang = sobel_grad(gray)

    deg = np.rad2deg(ang) % 180
    h, w = gray.shape

    n_rows = h // cell
    n_cols = w // cell

    feats = []
    bin_edges = np.linspace(0, 180, bins + 1)

    for i in range(n_rows):
        for j in range(n_cols):
            y1 = i * cell
            y2 = y1 + cell
            x1 = j * cell
            x2 = x1 + cell

            cell_ang = deg[y1:y2, x1:x2].ravel()
            cell_mag = mag[y1:y2, x1:x2].ravel()

            hist, _ = np.histogram(
                cell_ang,
                bins=bin_edges,
                weights=cell_mag,
            )

            feats.extend(hist)

    return norm_vec(np.array(feats, dtype=np.float32))


def lbp_feature(gray):
    c = gray[1:-1, 1:-1]

    code = np.zeros_like(c, dtype=np.uint8)

    code |= ((gray[:-2, :-2] >= c).astype(np.uint8) << 7)
    code |= ((gray[:-2, 1:-1] >= c).astype(np.uint8) << 6)
    code |= ((gray[:-2, 2:] >= c).astype(np.uint8) << 5)
    code |= ((gray[1:-1, 2:] >= c).astype(np.uint8) << 4)
    code |= ((gray[2:, 2:] >= c).astype(np.uint8) << 3)
    code |= ((gray[2:, 1:-1] >= c).astype(np.uint8) << 2)
    code |= ((gray[2:, :-2] >= c).astype(np.uint8) << 1)
    code |= ((gray[1:-1, :-2] >= c).astype(np.uint8) << 0)

    hist, _ = np.histogram(code.ravel(), bins=256, range=(0, 256))

    return norm_vec(hist.astype(np.float32))


def color_feature(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    h_hist, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 180))
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=8, range=(0, 256))
    v_hist, _ = np.histogram(hsv[:, :, 2], bins=8, range=(0, 256))

    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)

    return norm_vec(feat)

def weighted_concat(feats, weights):
    parts = []

    for name, weight in weights:
        parts.append(feats[name] * weight)

    return norm_vec(np.concatenate(parts))

def extract_feature_dict(path):
    bgr, gray = read_image(path)

    feats = {
        "edge": edge_feature(gray),
        "harris": harris_feature(gray),
        "hog": hog_feature(gray),
        "lbp": lbp_feature(gray),
        "color": color_feature(bgr),
    }

    feats["hog_lbp"] = norm_vec(
        np.concatenate(
            [
                feats["hog"],
                feats["lbp"],
            ]
        )
    )

    feats["hog_lbp_edge"] = norm_vec(
        np.concatenate(
            [
                feats["hog"],
                feats["lbp"],
                feats["edge"],
            ]
        )
    )

    feats["shape_texture_color"] = norm_vec(
        np.concatenate(
            [
                feats["hog"],
                feats["lbp"],
                feats["color"],
            ]
        )
    )

    feats["fusion_all"] = norm_vec(
        np.concatenate(
            [
                feats["edge"],
                feats["harris"],
                feats["hog"],
                feats["lbp"],
                feats["color"],
            ]
        )
    )

    feats["weighted_fusion"] = weighted_concat(
        feats,
        [
            ("hog", 1.2),
            ("lbp", 1.0),
            ("edge", 0.8),
            ("color", 0.5),
            ("harris", 0.3),
        ],
    )

    # 最终推荐融合版本：形状 + 纹理 + 边缘
    feats["fusion"] = feats["hog_lbp_edge"]

    return feats