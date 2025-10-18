from train_real_data import extract_6_dishes_to_frame_folders
from config import ROI_BOXES as roi_boxes
from config import DISH_TO_CLASS as dish_to_class
extract_6_dishes_to_frame_folders(
    "L_RL_EtOH_6.5_2025_09_25_49_2.mov",
    "data/",
    num_frames=200,
    roi_boxes=roi_boxes,
    dish_to_class=dish_to_class
)