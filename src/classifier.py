import numpy as np


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-8

    train_z = (train_x - mean) / std
    test_z = (test_x - mean) / std

    return train_z, test_z


def predict_knn(train_x, train_y, test_x, k=3):
    preds = []

    for x in test_x:
        d = np.linalg.norm(train_x - x, axis=1)
        idx = np.argsort(d)[:k]

        near_labels = train_y[idx]

        values, counts = np.unique(near_labels, return_counts=True)
        max_count = counts.max()
        candidates = values[counts == max_count]

        if len(candidates) == 1:
            pred = candidates[0]
        else:
            pred = near_labels[0]

        preds.append(pred)

    return np.array(preds, dtype=np.int64)