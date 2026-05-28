import cv2
import numpy as np

from config import IMG_SIZE


def norm_vec(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    n = np.linalg.norm(x) + 1e-8
    return x / n


def filt3(img, kernel):
    h, w = img.shape
    pad = np.pad(img, ((1, 1), (1, 1)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)

    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * pad[i:i + h, j:j + w]

    return out


def read_image(path):
    bgr = cv2.imread(str(path))

    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")

    bgr = cv2.resize(bgr, IMG_SIZE)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    return bgr, gray


def sobel_grad(gray):
    kx = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float32,
    )

    ky = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=np.float32,
    )

    gx = filt3(gray, kx)
    gy = filt3(gray, ky)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx)

    return gx, gy, mag, ang


def grid_mean(arr, rows, cols):
    h, w = arr.shape
    ch = h // rows
    cw = w // cols

    arr = arr[:rows * ch, :cols * cw]
    arr = arr.reshape(rows, ch, cols, cw)

    return arr.mean(axis=(1, 3)).ravel()