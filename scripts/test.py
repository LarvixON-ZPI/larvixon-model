import sys

sys.path.append(".")
from scripts.train_real_data import extract_8_dishes_to_frame_folders
from src.config import ROI_BOXES as roi_boxes
from src.config import DISH_TO_CLASS as dish_to_class

extract_8_dishes_to_frame_folders(
    "internal_data/L_RL_2025_09_22_48.mov",
    "data/",
    num_frames=50,
    dish_to_class=dish_to_class,
)
