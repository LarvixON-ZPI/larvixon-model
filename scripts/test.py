import sys

sys.path.append(".")
from scripts.train_real_data import extract_8_dishes_to_frame_folders
from src.config import ROI_BOXES as roi_boxes
from src.config import DISH_TO_CLASS as dish_to_class

extract_8_dishes_to_frame_folders(
    "internal_data/output.mp4",
    "data/",
    num_frames=15,
    dish_to_class=dish_to_class,
)
