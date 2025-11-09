import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.utils.logger import logger


class LarvaeDetector:
    """
    Detects larvae presence and movement within petri dish ROIs.
    Enhances existing motion detection to work with automatically detected
    dishes.
    """

    def __init__(
        self,
        background_history: int = 500,
        var_threshold: float = 16,
        detect_shadows: bool = False,
    ):
        """
        Initialize the larvae detector.

        Parameters:
        - background_history: Number of frames for background learning
        - var_threshold: Threshold for background subtraction
        - detect_shadows: Whether to detect shadows
        """
        self.background_history = background_history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows

    def create_background_subtractor(self):
        """Create and return a background subtractor."""
        return cv2.createBackgroundSubtractorMOG2(
            history=self.background_history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

    def detect_larvae_appearance(
        self,
        video_path: str,
        dish_rois: List[Tuple[int, int, int, int]],
        max_frames_check: int = 500,
        min_motion_area: int = 100,
        max_motion_area: int = 5000,
        sustain_frames: int = 5,
    ) -> List[int]:
        """
        Detect when larvae first appear in each petri dish.

        Parameters:
        - video_path: Path to video file
        - dish_rois: List of dish ROIs [(x, y, w, h), ...]
        - max_frames_check: Maximum frames to check for larvae appearance
        - min_motion_area: Minimum area of motion to consider as larvae
        - max_motion_area: Maximum area to avoid detecting hands
        - sustain_frames: Frames of sustained motion required

        Returns:
        - List of frame indices where larvae first appear in each dish
        """
        logger.info(
            f"Detecting larvae appearance in {len(dish_rois)} dishes"
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        bg_subtractors = [
            self.create_background_subtractor() for _ in dish_rois
        ]

        motion_streaks = [0] * len(dish_rois)
        first_appearance = [-1] * len(dish_rois)
        frame_idx = 0
        while frame_idx < max_frames_check:
            ret, frame = cap.read()
            if not ret:
                break

            for dish_idx, (x, y, w, h) in enumerate(dish_rois):
                if first_appearance[dish_idx] != -1:
                    continue

                roi_frame = frame[y : y + h, x : x + w]
                if roi_frame.size == 0:
                    continue

                fg_mask = bg_subtractors[dish_idx].apply(roi_frame)

                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (3, 3)
                )
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(
                    fg_mask, cv2.MORPH_CLOSE, kernel
                )

                contours, _ = cv2.findContours(
                    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                total_motion_area = sum(
                    cv2.contourArea(c) for c in contours
                )

                if min_motion_area <= total_motion_area <= max_motion_area:
                    motion_streaks[dish_idx] += 1

                    if motion_streaks[dish_idx] >= sustain_frames:
                        first_appearance[dish_idx] = max(
                            0, frame_idx - sustain_frames
                        )
                        logger.info(
                            f"Larvae detected in dish {dish_idx} "
                            f"at frame {first_appearance[dish_idx]}"
                        )
                else:
                    motion_streaks[dish_idx] = 0

            frame_idx += 1

            if all(appearance != -1 for appearance in first_appearance):
                break

        cap.release()

        for i in range(len(first_appearance)):
            if first_appearance[i] == -1:
                first_appearance[i] = 0
                logger.warning(
                    f"No larvae detected in dish {i}, "
                    f"starting from frame 0"
                )

        return first_appearance

    def validate_larvae_presence(
        self,
        video_path: str,
        dish_roi: Tuple[int, int, int, int],
        start_frame: int,
        num_frames_check: int = 50,
    ) -> bool:
        """
        Validate that larvae are actually present in a dish starting from
        a specific frame.

        Parameters:
        - video_path: Path to video file
        - dish_roi: Single dish ROI (x, y, w, h)
        - start_frame: Frame to start checking from
        - num_frames_check: Number of frames to analyze

        Returns:
        - True if larvae presence is confirmed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        x, y, w, h = dish_roi
        bg_subtractor = self.create_background_subtractor()

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        motion_frames = 0
        total_frames = 0

        for i in range(num_frames_check):
            ret, frame = cap.read()
            if not ret:
                break

            roi_frame = frame[y : y + h, x : x + w]
            if roi_frame.size == 0:
                continue

            fg_mask = bg_subtractor.apply(roi_frame)

            motion_pixels = cv2.countNonZero(fg_mask)

            if motion_pixels > 50:  # Some motion detected
                motion_frames += 1

            total_frames += 1

        cap.release()

        motion_ratio = motion_frames / max(total_frames, 1)
        return motion_ratio > 0.2

    def get_quality_frames(
        self,
        video_path: str,
        dish_roi: Tuple[int, int, int, int],
        start_frame: int,
        end_frame: int,
        target_frames: int = 225,
    ) -> List[int]:
        """
        Select high-quality frames from a dish ROI for model training.

        Parameters:
        - video_path: Path to video file
        - dish_roi: Single dish ROI (x, y, w, h)
        - start_frame: Start frame for extraction
        - end_frame: End frame for extraction
        - target_frames: Number of frames to select

        Returns:
        - List of frame indices with good quality
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        x, y, w, h = dish_roi
        frame_qualities = []

        total_frames = end_frame - start_frame
        if total_frames <= target_frames:
            return list(range(start_frame, end_frame))

        sample_step = max(1, total_frames // (target_frames * 2))
        sample_frames = list(range(start_frame, end_frame, sample_step))

        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            roi_frame = frame[y : y + h, x : x + w]
            if roi_frame.size == 0:
                continue

            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            contrast = np.std(gray)

            brightness = np.mean(gray)

            quality_score = sharpness * contrast

            if brightness < 30 or brightness > 220:
                quality_score *= 0.5

            frame_qualities.append((frame_idx, quality_score))

        cap.release()

        frame_qualities.sort(key=lambda x: x[1], reverse=True)
        selected_frames = [
            idx for idx, _ in frame_qualities[:target_frames]
        ]
        selected_frames.sort()

        return selected_frames


def detect_larvae_in_dishes(
    video_path: str,
    dish_rois: List[Tuple[int, int, int, int]],
    max_frames_check: int = 500,
) -> List[int]:
    """
    Convenience function to detect larvae appearance in multiple dishes.

    Parameters:
    - video_path: Path to video file
    - dish_rois: List of dish ROIs
    - max_frames_check: Maximum frames to check

    Returns:
    - List of frame indices where larvae first appear
    """
    detector = LarvaeDetector()
    return detector.detect_larvae_appearance(
        video_path, dish_rois, max_frames_check
    )
