import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import os
from src.utils.logger import logger


class GridBasedPetriDetector:
    """
    Grid-based petri dish detector designed for videos with dishes arranged
    in a regular grid pattern. More robust than contour-based detection.
    """

    def __init__(
        self,
        grid_rows: int = 4,
        grid_cols: int = 2,
        padding_ratio: float = 0.1,
        min_variance_threshold: float = 100.0,
    ):
        """
        Initialize grid-based detector.

        Parameters:
        - grid_rows: Number of rows in the grid
        - grid_cols: Number of columns in the grid
        - padding_ratio: Padding around each grid cell (0.1 = 10%)
        - min_variance_threshold: Minimum variance to consider a region valid
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.padding_ratio = padding_ratio
        self.min_variance_threshold = min_variance_threshold

    def detect_grid_layout(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect petri dishes using grid layout assumption.

        Parameters:
        - frame: Input frame

        Returns:
        - List of detected dish ROIs (x, y, w, h)
        """
        height, width = frame.shape[:2]

        cell_width = width // self.grid_cols
        cell_height = height // self.grid_rows

        pad_w = int(cell_width * self.padding_ratio)
        pad_h = int(cell_height * self.padding_ratio)

        detected_rois = []

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x = col * cell_width
                y = row * cell_height

                roi_x = x + pad_w
                roi_y = y + pad_h
                roi_w = cell_width - 2 * pad_w
                roi_h = cell_height - 2 * pad_h

                roi_x = max(0, roi_x)
                roi_y = max(0, roi_y)
                roi_w = min(roi_w, width - roi_x)
                roi_h = min(roi_h, height - roi_y)

                if roi_w > 0 and roi_h > 0:
                    if self._validate_roi_content(
                        frame, (roi_x, roi_y, roi_w, roi_h)
                    ):
                        detected_rois.append((roi_x, roi_y, roi_w, roi_h))

        logger.info(
            f"Grid-based detection found {len(detected_rois)} valid regions"
        )
        return detected_rois

    def _validate_roi_content(
        self, frame: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> bool:
        """
        Validate that ROI contains meaningful content (not empty space).

        Parameters:
        - frame: Input frame
        - roi: ROI to validate (x, y, w, h)

        Returns:
        - True if ROI contains meaningful content
        """
        x, y, w, h = roi

        roi_frame = frame[y : y + h, x : x + w]

        if roi_frame.size == 0:
            return False

        if len(roi_frame.shape) == 3:
            gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi_frame

        variance = np.var(gray_roi.astype(np.float32))

        if variance < self.min_variance_threshold:
            return False
            
        return self._check_for_dish_patterns(gray_roi)

    def _check_for_dish_patterns(self, gray_roi: np.ndarray) -> bool:
        """
        Look for patterns that suggest presence of a petri dish.

        Parameters:
        - gray_roi: Grayscale ROI image

        Returns:
        - True if dish-like patterns are detected
        """
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(gray_roi.shape) * 0.3),
            param1=50,
            param2=30,
            minRadius=int(min(gray_roi.shape) * 0.1),
            maxRadius=int(min(gray_roi.shape) * 0.8),
        )

        if circles is not None and len(circles[0]) > 0:
            return True

        edges = cv2.Canny(blurred, 30, 100)
        edge_density = np.count_nonzero(edges) / edges.size

        if edge_density > 0.02: 
            return True

        h, w = gray_roi.shape
        center_region = gray_roi[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_var = np.var(center_region.astype(np.float32))
        total_var = np.var(gray_roi.astype(np.float32))

        if abs(center_var - total_var) > 50:
            return True

        return True 


class AdaptiveGridDetector:
    """
    Adaptive detector that tries multiple grid configurations and
    selects the best one based on content analysis.
    """

    def __init__(self):
        self.grid_configs = [
            {"rows": 4, "cols": 2, "name": "4x2_vertical"},
            {"rows": 2, "cols": 4, "name": "2x4_horizontal"},
            {"rows": 3, "cols": 3, "name": "3x3_square"},
            {"rows": 2, "cols": 3, "name": "2x3_mixed"},
            {"rows": 1, "cols": 8, "name": "1x8_line"},
        ]

    def detect_best_layout(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], str]:
        """
        Try multiple grid configurations and return the best one.

        Parameters:
        - frame: Input frame

        Returns:
        - Tuple of (best_detections, config_name)
        """
        best_detections = []
        best_score = 0
        best_config_name = "none"

        for config in self.grid_configs:
            detector = GridBasedPetriDetector(
                grid_rows=config["rows"],
                grid_cols=config["cols"],
                padding_ratio=0.15, 
            )

            detections = detector.detect_grid_layout(frame)
            score = self._score_detections(frame, detections)

            logger.info(
                f"Grid {config['name']}: {len(detections)} detections, score: {score:.2f}"
            )

            if score > best_score:
                best_score = score
                best_detections = detections
                best_config_name = config["name"]

        logger.info(
            f"Best grid configuration: {best_config_name} with {len(best_detections)} detections"
        )
        return best_detections, best_config_name

    def _score_detections(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, int, int, int]],
    ) -> float:
        """
        Score a set of detections based on various criteria.

        Parameters:
        - frame: Input frame
        - detections: List of detected ROIs

        Returns:
        - Score (higher is better)
        """
        if not detections:
            return 0.0

        total_score = 0.0

        for roi in detections:
            x, y, w, h = roi
            roi_frame = frame[y : y + h, x : x + w]

            if roi_frame.size == 0:
                continue

            if len(roi_frame.shape) == 3:
                gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi_frame

            variance = np.var(gray_roi.astype(np.float32))
            variance_score = min(variance / 1000, 1.0)

            area = w * h
            size_score = min(area / 50000, 1.0)  

            aspect_ratio = w / h
            ar_score = 1.0 - abs(
                aspect_ratio - 1.0
            )  
            ar_score = max(0, ar_score)

            roi_score = (variance_score + size_score + ar_score) / 3
            total_score += roi_score

        return total_score / len(detections)


def detect_dishes_with_grid(
    frame: np.ndarray,
    save_visualization: bool = True,
    output_path: str = "grid_detection_result.jpg",
) -> List[Tuple[int, int, int, int]]:
    """
    Convenience function to detect dishes using grid-based approach.

    Parameters:
    - frame: Input frame
    - save_visualization: Whether to save visualization
    - output_path: Path for visualization image

    Returns:
    - List of detected ROIs
    """
    detector = AdaptiveGridDetector()
    detections, config_name = detector.detect_best_layout(frame)

    if save_visualization:
        vis_frame = frame.copy()

        for i, (x, y, w, h) in enumerate(detections):
            cv2.rectangle(
                vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 3
            )
            cv2.putText(
                vis_frame,
                f"Dish {i+1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            vis_frame,
            f"Grid: {config_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            vis_frame,
            f"Detections: {len(detections)}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )

        cv2.imwrite(output_path, vis_frame)
        logger.info(f"Grid detection visualization saved to {output_path}")

    return detections


def detect_dishes_in_video_grid(
    video_path: str,
    frame_index: int = 0,
    save_detection_image: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Grid-based detection for video files - replacement for contour detection.

    Parameters:
    - video_path: Path to video file
    - frame_index: Frame to analyze
    - save_detection_image: Whether to save visualization

    Returns:
    - List of detected ROIs
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read frame {frame_index} from video")

    if save_detection_image:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = (
            f"grid_detection_{video_name}_frame_{frame_index}.jpg"
        )
    else:
        output_path = None

    return detect_dishes_with_grid(
        frame, save_detection_image, output_path
    )
