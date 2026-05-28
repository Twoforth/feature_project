import numpy as np

from config import (
    TRAIN_DIR,
    TEST_DIR,
    OUT_DIR,
    VIS_DIR,
    K_NEIGHBORS,
    TOPK,
    FEATURE_NAMES,
    make_output_dirs,
)

from classifier import standardize, predict_knn

from evaluate import (
    collect_images,
    build_feature_cache,
    make_matrix,
    confusion_matrix,
)

from visualize import (
    plot_confusion,
    plot_accuracy_bar,
    save_sample_visuals,
    save_topk_retrieval,
)


def save_retrieval_text(query_path, result_paths, distances, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Top-K retrieval result\n")
        f.write("======================\n")
        f.write(f"Query image: {query_path.name}\n")
        f.write(f"Query class: {query_path.parent.name}\n\n")

        for i, path in enumerate(result_paths):
            f.write(f"Top-{i + 1}: {path.name}\n")
            f.write(f"Class: {path.parent.name}\n")
            f.write(f"Distance: {distances[i]:.4f}\n\n")


def choose_retrieval_example(best_train_x, best_test_x, test_y, best_pred, test_paths, target_class="Motorbikes"):
    correct_ids = np.where(best_pred == test_y)[0]

    if len(correct_ids) == 0:
        return 0

    target_ids = [
        idx for idx in correct_ids
        if test_paths[idx].parent.name == target_class
    ]

    if len(target_ids) == 0:
        target_ids = correct_ids

    best_id = target_ids[0]
    best_dist = float("inf")

    for idx in target_ids:
        d = np.linalg.norm(best_train_x - best_test_x[idx], axis=1)
        min_d = d.min()

        if min_d < best_dist:
            best_dist = min_d
            best_id = idx

    return int(best_id)

def row_norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def make_weighted_fusion_matrix(train_cache, test_cache):
    weights = [
        ("hog", 1.00),
        ("lbp", 0.25),
        ("edge", 0.12),
        ("color", 0.05),
        ("harris", 0.03),
    ]

    train_parts = []
    test_parts = []

    for name, weight in weights:
        train_x = make_matrix(train_cache, name)
        test_x = make_matrix(test_cache, name)

        train_x, test_x = standardize(train_x, test_x)

        train_parts.append(train_x * weight)
        test_parts.append(test_x * weight)

    train_fusion = np.hstack(train_parts)
    test_fusion = np.hstack(test_parts)

    train_fusion = row_norm(train_fusion)
    test_fusion = row_norm(test_fusion)

    return train_fusion, test_fusion

def run_experiment():
    make_output_dirs()

    train_paths, train_y, train_classes = collect_images(TRAIN_DIR)
    test_paths, test_y, test_classes = collect_images(TEST_DIR)

    if train_classes != test_classes:
        raise ValueError("Train and test classes are not the same.")

    print("Classes:", train_classes)
    print("Train images:", len(train_paths))
    print("Test images:", len(test_paths))

    if len(train_paths) == 0 or len(test_paths) == 0:
        raise ValueError("No images found. Please check data/selected/train and data/selected/test.")

    save_sample_visuals(train_paths[0])

    train_cache = build_feature_cache(train_paths)
    test_cache = build_feature_cache(test_paths)

    results = {}

    best_pred = None
    best_name = None
    best_acc = -1

    best_train_x = None
    best_test_x = None

    for key in FEATURE_NAMES:
        print(f"\nRunning kNN with feature: {key}")

        if key == "weighted_fusion":
            train_x, test_x = make_weighted_fusion_matrix(train_cache, test_cache)
        else:
            train_x = make_matrix(train_cache, key)
            test_x = make_matrix(test_cache, key)

            train_x, test_x = standardize(train_x, test_x)

        pred = predict_knn(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            k=K_NEIGHBORS,
        )

        acc = (pred == test_y).mean()
        results[key] = acc

        print(f"{key} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_pred = pred
            best_name = key
            best_train_x = train_x.copy()
            best_test_x = test_x.copy()

    cm = confusion_matrix(test_y, best_pred, len(train_classes))

    print(f"\nBest feature: {best_name}, accuracy: {best_acc:.4f}")

    plot_confusion(
        cm,
        train_classes,
        OUT_DIR / "confusion_matrix_best.png",
    )

    plot_accuracy_bar(
        results,
        OUT_DIR / "feature_accuracy_comparison.png",
    )

    query_id = choose_retrieval_example(
        best_train_x=best_train_x,
        best_test_x=best_test_x,
        test_y=test_y,
        best_pred=best_pred,
        test_paths=test_paths,
        target_class="Motorbikes",
    )

    distances = np.linalg.norm(best_train_x - best_test_x[query_id], axis=1)
    top_idx = np.argsort(distances)[:TOPK]

    result_paths = [train_paths[i] for i in top_idx]
    top_distances = distances[top_idx]

    save_topk_retrieval(
        query_path=test_paths[query_id],
        result_paths=result_paths,
        distances=top_distances,
        save_path=OUT_DIR / "topk_retrieval_best.png",
    )

    save_retrieval_text(
        query_path=test_paths[query_id],
        result_paths=result_paths,
        distances=top_distances,
        save_path=OUT_DIR / "topk_retrieval_result.txt",
    )

    with open(OUT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("Feature accuracy results\n")
        f.write("========================\n")

        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\n")
        f.write(f"best_feature: {best_name}\n")
        f.write(f"best_accuracy: {best_acc:.4f}\n")

    print("\nDone.")
    print("Results saved to:", OUT_DIR)
    print("Feature maps saved to:", VIS_DIR)
    print("Top-K retrieval figure saved to:", OUT_DIR / "topk_retrieval_best.png")


if __name__ == "__main__":
    run_experiment()