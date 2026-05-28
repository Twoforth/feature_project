from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "data" / "selected" / "train"
TEST_DIR = ROOT / "data" / "selected" / "test"

OUT_DIR = ROOT / "outputs"
VIS_DIR = OUT_DIR / "feature_maps"

IMG_SIZE = (128, 128)
K_NEIGHBORS = 3
TOPK = 5

FEATURE_NAMES = [
    "edge",
    "harris",
    "hog",
    "lbp",
    "color",
    "hog_lbp",
    "hog_lbp_edge",
    "shape_texture_color",
    "fusion_all",
    "weighted_fusion",
    "fusion",
]


def make_output_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    VIS_DIR.mkdir(exist_ok=True)