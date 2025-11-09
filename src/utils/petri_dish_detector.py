import cv2
import numpy as np
from typing import List, Tuple, Optional
import json
import os
from src.utils.logger import logger


class PetriDishDetector:
    """
    Detects square/rectangular petri dishes with rounded corners in video
    frames. Provides automatic ROI detection to replace manual selection.
    """

    def __init__(
        self,
        min_area: int = 10000,
        max_area: int = 200000,
        min_aspect_ratio: float = 0.7,
        max_aspect_ratio: float = 1.4,
        min_extent: float = 0.6,
    ):
        """
        Initialize the petri dish detector.

        Parameters:
        - min_area: Minimum area of detected dishes in pixels
        - max_area: Maximum area of detected dishes in pixels
        - min_aspect_ratio: Minimum width/height ratio for dishes
        - max_aspect_ratio: Maximum width/height ratio for dishes
        - min_extent: Minimum extent (area/bounding_rect_area) for rounded
          shapes
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_extent = min_extent

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better edge detection.

        Parameters:
        - frame: Input BGR frame

        Returns:
        - Preprocessed grayscale frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        blurred = cv2.GaussianBlur(filtered, (5, 5), 0)

        return blurred

    def detect_edges(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        Detect edges in the preprocessed frame.

        Parameters:
        - gray_frame: Preprocessed grayscale frame

        Returns:
        - Binary edge image
        """

        adaptive_thresh = cv2.adaptiveThreshold(
            gray_frame,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        edges = cv2.Canny(gray_frame, 50, 150, apertureSize=3)

        combined = cv2.bitwise_or(edges, adaptive_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        return closed

    def filter_contours(self, contours) -> List[np.ndarray]:
        """
        Filter contours to find potential petri dishes.

        Parameters:
        - contours: List of detected contours

        Returns:
        - Filtered list of contours that could be petri dishes
        """
        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            if (
                aspect_ratio < self.min_aspect_ratio
                or aspect_ratio > self.max_aspect_ratio
            ):
                continue

            rect_area = w * h
            extent = float(area) / rect_area

            if extent < self.min_extent:
                continue

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 4 or len(approx) > 12:
                continue

            valid_contours.append(contour)

        return valid_contours

    def validate_dish_appearance(
        self, frame: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> bool:
        """
        Validate that the ROI actually looks like a petri dish.

        Parameters:
        - frame: Original frame
        - roi: Region of interest (x, y, w, h)

        Returns:
        - True if the ROI looks like a petri dish
        """
        x, y, w, h = roi

        roi_frame = frame[y : y + h, x : x + w]

        if roi_frame.size == 0:
            return False

        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        std_dev = float(np.std(gray_roi.astype(np.float32)))

        min_val = np.min(gray_roi)
        max_val = np.max(gray_roi)
        contrast = max_val - min_val

        if std_dev < 10:
            return False

        if contrast < 30:
            return False

        return True

    def detect_dishes(
        self,
        frame: np.ndarray,
        validate_appearance: bool = True,
        min_distance: int = 50,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect petri dishes in a frame.

        Parameters:
        - frame: Input BGR frame
        - validate_appearance: Whether to validate dish appearance
        - min_distance: Minimum distance between detected dishes

        Returns:
        - List of bounding rectangles (x, y, w, h) for detected dishes
        """
        logger.info("Starting petri dish detection...")

        gray = self.preprocess_frame(frame)

        edges = self.detect_edges(gray)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        logger.info(f"Found {len(contours)} total contours")

        valid_contours = self.filter_contours(list(contours))

        logger.info(
            f"Found {len(valid_contours)} valid contours after filtering"
        )

        detected_dishes = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)

            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)

            roi = (x, y, w, h)

            if validate_appearance and not self.validate_dish_appearance(
                frame, roi
            ):
                logger.debug(
                    f"Rejected ROI {roi} due to appearance " f"validation"
                )
                continue

            detected_dishes.append(roi)

        filtered_dishes = self.remove_overlapping_detections(
            detected_dishes, min_distance
        )

        logger.info(
            f"Final detection result: {len(filtered_dishes)} petri dishes"
        )

        return filtered_dishes

    def remove_overlapping_detections(
        self,
        detections: List[Tuple[int, int, int, int]],
        min_distance: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Remove overlapping or too-close detections.

        Parameters:
        - detections: List of bounding rectangles
        - min_distance: Minimum distance between centers

        Returns:
        - Filtered list of detections
        """
        if len(detections) <= 1:
            return detections

        centers = []
        for x, y, w, h in detections:
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))

        keep = []
        for i, (center, detection) in enumerate(zip(centers, detections)):
            too_close = False
            for j in keep:
                other_center = centers[j]
                distance = np.sqrt(
                    (center[0] - other_center[0]) ** 2
                    + (center[1] - other_center[1]) ** 2
                )
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                keep.append(i)

        return [detections[i] for i in keep]

    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, int, int, int]],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize detected petri dishes on the frame.

        Parameters:
        - frame: Original frame
        - detections: List of detected dish ROIs
        - save_path: Optional path to save visualization

        Returns:
        - Frame with visualization overlay
        """
        vis_frame = frame.copy()

        for i, (x, y, w, h) in enumerate(detections):
            cv2.rectangle(
                vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            label = f"Dish {i}"
            cv2.putText(
                vis_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if save_path:
            cv2.imwrite(save_path, vis_frame)
            logger.info(f"Saved visualization to {save_path}")

        return vis_frame


def detect_dishes_in_video(
    video_path: str,
    frame_index: int = 0,
    save_detection_image: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Convenience function to detect petri dishes in a specific video frame.

    Parameters:
    - video_path: Path to video file
    - frame_index: Which frame to analyze (default: first frame)
    - save_detection_image: Whether to save detection visualization

    Returns:
    - List of detected dish ROIs (x, y, w, h)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read frame {frame_index} from video")

    detector = PetriDishDetector()

    detections = detector.detect_dishes(frame)

    if save_detection_image:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        vis_path = f"petri_detection_{video_name}_frame_{frame_index}.jpg"
        detector.visualize_detections(frame, detections, vis_path)

    return detections


def save_detections_to_json(
    detections: List[Tuple[int, int, int, int]],
    output_path: str = "detected_roi_boxes.json",
):
    """
    Save detected ROIs to JSON file in the same format as roi_boxes.json.

    Parameters:
    - detections: List of detected dish ROIs
    - output_path: Output JSON file path
    """
    with open(output_path, "w") as f:
        json.dump(detections, f, indent=2)

    logger.info(f"Saved {len(detections)} detected ROIs to {output_path}")
