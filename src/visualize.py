import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import VIS_DIR
from preprocess import read_image, sobel_grad, filt3


def plot_confusion(cm, classes, save_path):
    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(cm, cmap="Blues")

    plt.title("Confusion Matrix of Weighted Fusion", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    plt.xticks(range(len(classes)), classes, rotation=35, ha="right", fontsize=10)
    plt.yticks(range(len(classes)), classes, fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]

            if value > 0:
                plt.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="black",
                )

    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_accuracy_bar(results, save_path):
    display_names = {
        "edge": "Edge",
        "harris": "Harris",
        "hog": "HOG",
        "lbp": "LBP",
        "color": "Color",
        "hog_lbp": "HOG+LBP",
        "hog_lbp_edge": "HOG+LBP+Edge",
        "shape_texture_color": "Shape+Texture+Color",
        "fusion_all": "All Fusion",
        "weighted_fusion": "Weighted Fusion",
        "fusion": "Selected Fusion",
    }

    names = list(results.keys())
    labels = [display_names.get(k, k) for k in names]
    accs = [results[k] for k in names]

    plt.figure(figsize=(11, 5.5))
    bars = plt.bar(labels, accs)

    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy Comparison of Different Feature Representations", fontsize=14)

    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=10)

    for i, v in enumerate(accs):
        plt.text(
            i,
            v + 0.012,
            f"{v:.3f}",
            ha="center",
            fontsize=9,
        )

    best_idx = int(np.argmax(accs))
    bars[best_idx].set_hatch("//")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_sample_visuals(sample_path):
    bgr, gray = read_image(sample_path)

    _, _, mag, _ = sobel_grad(gray)

    th = mag.mean() + 0.5 * mag.std()
    edge = (mag > th).astype(np.uint8) * 255

    cv2.imwrite(str(VIS_DIR / "sample_original.jpg"), bgr)
    cv2.imwrite(str(VIS_DIR / "sample_gray.jpg"), (gray * 255).astype(np.uint8))
    cv2.imwrite(str(VIS_DIR / "sample_edge_sobel.jpg"), edge)

    gx, gy, _, _ = sobel_grad(gray)

    ixx = gx * gx
    iyy = gy * gy
    ixy = gx * gy

    box = np.ones((3, 3), dtype=np.float32) / 9.0

    sxx = filt3(ixx, box)
    syy = filt3(iyy, box)
    sxy = filt3(ixy, box)

    r = sxx * syy - sxy ** 2 - 0.04 * (sxx + syy) ** 2

    if r.max() > 0:
        corners = r > 0.01 * r.max()
    else:
        corners = np.zeros_like(r, dtype=bool)

    show = bgr.copy()
    pts = np.argwhere(corners)

    if len(pts) > 200:
        pts = pts[:200]

    for y, x in pts:
        cv2.circle(show, (int(x), int(y)), 1, (0, 0, 255), -1)

    cv2.imwrite(str(VIS_DIR / "sample_harris_corners.jpg"), show)

def save_topk_retrieval(query_path, result_paths, distances, save_path):
    all_paths = [query_path] + result_paths
    titles = ["Query"] + [f"Top-{i}" for i in range(1, len(result_paths) + 1)]

    plt.figure(figsize=(10, 6))

    for i, path in enumerate(all_paths):
        bgr = cv2.imread(str(path))

        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (180, 180))

        plt.subplot(2, 3, i + 1)
        plt.imshow(rgb)
        plt.axis("off")

        cls_name = path.parent.name

        if i == 0:
            title = f"{titles[i]}\n{cls_name}"
        else:
            title = f"{titles[i]}\n{cls_name}\nD={distances[i - 1]:.2f}"

        plt.title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()