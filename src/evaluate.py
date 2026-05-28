import os

import numpy as np

from features import extract_feature_dict


def collect_images(data_dir):
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    class_to_id = {name: i for i, name in enumerate(classes)}

    paths = []
    labels = []

    for cls in classes:
        cls_dir = data_dir / cls

        for name in os.listdir(cls_dir):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(cls_dir / name)
                labels.append(class_to_id[cls])

    return paths, np.array(labels, dtype=np.int64), classes


def build_feature_cache(paths):
    cache = []

    for i, path in enumerate(paths):
        print(f"Extracting features: {i + 1}/{len(paths)}  {path.name}")
        cache.append(extract_feature_dict(path))

    return cache


def make_matrix(cache, key):
    return np.vstack([item[key] for item in cache]).astype(np.float32)


def confusion_matrix(y_true, y_pred, n_cls):
    cm = np.zeros((n_cls, n_cls), dtype=np.int32)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm
