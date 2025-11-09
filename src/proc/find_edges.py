import cv2
import os


def find_edges_first_frame(video_path, output_image_name="first_frame_edges.jpg"):
    """
    Analyzes only the first frame of a video to find all edge contours
    and their bounding boxes (ROIs).

    Uses sensitive Canny settings to detect faint/thin edges.

    Args:
        video_path (str): The file path to the input video (e.g., ".mov").
        output_image_name (str): Filename to save the visualization.

    Returns:
        list: A list of tuples. Each tuple represents an ROI in 
              (x, y, w, h) format.
              Returns None if the video cannot be opened or frame read.
    """

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    print("Reading the first frame...")
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read the first frame from the video.")
        cap.release()
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    low_thresh = 20
    high_thresh = 40
    print(f"Running Canny edge detection with thresholds: {low_thresh}, {high_thresh}")
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_rois = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        frame_rois.append((x, y, w, h))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imwrite(output_image_name, frame)
    print(f"\nProcessing complete. Found {len(frame_rois)} edge ROIs.")
    print(f"Visualization saved to '{output_image_name}'")

    cap.release()
    cv2.destroyAllWindows()

    return frame_rois


if __name__ == "__main__":

    VIDEO_FILE_PATH = "/home/coolka/projects/python/larvixon_model/internal_data/L_RL_2025_09_22_48.mov" 

    roi_data = find_edges_first_frame(VIDEO_FILE_PATH)

    if roi_data is not None:
        print(f"\n--- ROI Data for First Frame (Total: {len(roi_data)}) ---")
        print(roi_data[:10])
    else:
        print("Could not process the video.")