import os
import random
import shutil
from pathlib import Path

RAW_DIR = Path(r"F:\feature_project\data\raw\101_ObjectCategories")
OUT_DIR = Path(r"F:\feature_project\data\selected")

CLASSES = [
    "airplanes",
    "Motorbikes",
    "Faces_easy",
    "watch",
    "chair",
    "sunflower",
]

N_PER_CLASS = 50
N_TRAIN = 35
SEED = 42


def make_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def copy_images():
    random.seed(SEED)

    for cls in CLASSES:
        src_dir = RAW_DIR / cls

        if not src_dir.exists():
            print(f"[Warning] Class not found: {cls}")
            continue

        imgs = [
            src_dir / name
            for name in os.listdir(src_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(imgs)
        imgs = imgs[:N_PER_CLASS]

        train_imgs = imgs[:N_TRAIN]
        test_imgs = imgs[N_TRAIN:]

        train_out = OUT_DIR / "train" / cls
        test_out = OUT_DIR / "test" / cls

        make_dir(train_out)
        make_dir(test_out)

        for img in train_imgs:
            shutil.copy(img, train_out / img.name)

        for img in test_imgs:
            shutil.copy(img, test_out / img.name)

        print(f"{cls}: train={len(train_imgs)}, test={len(test_imgs)}")


if __name__ == "__main__":
    copy_images()