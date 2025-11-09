import cv2
import os
from src.utils.logger import logger


def find_shapes_first_frame(
    video_path, output_image_name="first_frame_shapes.jpg"
):
    """
    Analyzes the first frame of a video to find low-contrast shapes
    using adaptive thresholding on the red channel.

    Args:
        video_path (str): The file path to the input video (e.g., ".mov").
        output_image_name (str): Filename to save the visualization.

    Returns:
        list: A list of tuples for filtered ROIs in (x, y, w, h) format.
              Returns None if the video cannot be opened or frame read.
    """
    if not os.path.exists(video_path):
        logger.error(f"Error: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {video_path}")
        return None

    logger.info("Reading the first frame...")
    ret, frame = cap.read()
    if not ret:
        logger.error(
            "Error: Could not read the first frame from the video."
        )
        cap.release()
        return None

    b, g, r_channel = cv2.split(frame)

    block_size = 15
    sensitivity_const = 4

    logger.info(
        f"Applying adaptive threshold with BlockSize={block_size}, C={sensitivity_const}"
    )
    thresh_img = cv2.adaptiveThreshold(
        r_channel,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        sensitivity_const,
    )

    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    frame_rois = []

    logger.info(f"Found {len(contours)} raw contours. Filtering...")
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 50 and h > 50:
            frame_rois.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_image_name, frame)
    debug_img_name = "first_frame_DEBUG_threshold.jpg"
    cv2.imwrite(debug_img_name, thresh_img)

    logger.info(
        f"\nProcessing complete. Found {len(frame_rois)} filtered ROIs."
    )
    logger.info(f"Visualization saved to: '{output_image_name}'")
    logger.info(
        f"Debug view saved to:    '{debug_img_name}' (Check this to tune!)"
    )

    cap.release()
    cv2.destroyAllWindows()

    return frame_rois


if __name__ == "__main__":

    VIDEO_FILE_PATH = "/home/coolka/projects/python/larvixon_model/internal_data/L_RL_2025_09_22_48.mov"

    roi_data = find_shapes_first_frame(VIDEO_FILE_PATH)

    if roi_data is not None:
        logger.info(
            f"\n--- ROI Data for First Frame (Total: {len(roi_data)}) ---"
        )
        logger.info(roi_data)
    else:
        logger.error("Could not process the video.")
