import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from src.utils.logger import logger
from src.utils.petri_dish_detector import (
    PetriDishDetector,
    detect_dishes_in_video,
)
from src.utils.grid_based_detector import detect_dishes_in_video_grid
from src.utils.larvae_detector import LarvaeDetector
import src.config as config


class SmartVideoExtractor:
    """
    Smart extraction pipeline that automatically detects petri dishes,
    identifies larvae presence, and extracts high-quality training frames.
    """

    def __init__(self, use_detected_rois: bool = True):
        """
        Initialize the smart extractor.

        Parameters:
        - use_detected_rois: Whether to use automatic detection or manual ROIs
        """
        self.use_detected_rois = use_detected_rois
        self.dish_detector = PetriDishDetector()
        self.larvae_detector = LarvaeDetector()

    def extract_video_intelligently(
        self,
        video_path: str,
        output_dir: str,
        num_frames: int = 225,
        dish_to_class: Optional[Dict[int, str]] = None,
        detection_frame: int = 0,
        max_larvae_check_frames: int = 500,
    ) -> int:
        """
        Extract frames from video using intelligent detection and selection.

        Parameters:
        - video_path: Path to input video
        - output_dir: Directory to save extracted frames
        - num_frames: Number of frames to extract per dish
        - dish_to_class: Mapping from dish index to class name
        - detection_frame: Frame to use for dish detection
        - max_larvae_check_frames: Max frames to check for larvae

        Returns:
        - Number of sequences created
        """
        logger.info(f"Starting intelligent extraction for {video_path}")

        if self.use_detected_rois:
            logger.info("Using automatic petri dish detection...")
            try:
                dish_rois = detect_dishes_in_video(video_path, detection_frame, save_detection_image=True)
            except Exception as e:
                logger.warning(f"Contour-based detection failed: {e}")
                dish_rois = []

            # Fallback to grid-based detection if contour detection fails
            if not dish_rois:
                logger.info("Falling back to grid-based detection...")
                try:
                    dish_rois = detect_dishes_in_video_grid(
                        video_path,
                        detection_frame,
                        save_detection_image=True,
                    )
                except Exception as e:
                    logger.warning(f"Grid-based detection failed: {e}")
                    dish_rois = []
        else:
            logger.info("Using manual ROI boxes...")
            dish_rois = config.ROI_BOXES

        if not dish_rois:
            logger.error("No petri dishes detected or configured!")
            return 0

        logger.info(f"Found {len(dish_rois)} petri dishes")

        if dish_to_class is None:
            dish_to_class = config.DISH_TO_CLASS

        logger.info("Detecting larvae appearance...")
        larvae_start_frames = self.larvae_detector.detect_larvae_appearance(video_path, dish_rois, max_larvae_check_frames)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        sequences_created = 0

        for dish_idx, (dish_roi, start_frame) in enumerate(zip(dish_rois, larvae_start_frames)):

            if dish_idx not in dish_to_class:
                logger.warning(f"No class mapping for dish {dish_idx}, skipping")
                continue

            class_name = dish_to_class[dish_idx]

            seq_dir = os.path.join(
                output_dir,
                class_name,
                f"frames_{video_stem}_dish{dish_idx}",
            )
            os.makedirs(seq_dir, exist_ok=True)

            end_frame = total_frames

            quality_frames = self.larvae_detector.get_quality_frames(video_path, dish_roi, start_frame, end_frame, num_frames)

            if not quality_frames:
                logger.warning(f"No quality frames found for dish {dish_idx}")
                continue

            extracted_count = self._extract_frames_from_indices(video_path, dish_roi, quality_frames, seq_dir)

            if extracted_count > 0:
                sequences_created += 1
                logger.info(f"Created sequence for dish {dish_idx} " f"({class_name}): {extracted_count} frames")

        logger.info(f"Intelligent extraction complete: " f"{sequences_created} sequences created")
        return sequences_created

    def _extract_frames_from_indices(
        self,
        video_path: str,
        dish_roi: Tuple[int, int, int, int],
        frame_indices: List[int],
        output_dir: str,
    ) -> int:
        """
        Extract specific frames from video and save to directory.

        Parameters:
        - video_path: Path to video file
        - dish_roi: ROI for the dish (x, y, w, h)
        - frame_indices: List of frame indices to extract
        - output_dir: Directory to save frames

        Returns:
        - Number of frames successfully extracted
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return 0

        x, y, w, h = dish_roi
        extracted_count = 0

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                continue

            roi_frame = frame[y : y + h, x : x + w]

            if roi_frame.size == 0:
                logger.warning(f"Empty ROI for frame {frame_idx}")
                continue

            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            if cv2.imwrite(frame_path, roi_frame):
                extracted_count += 1
            else:
                logger.warning(f"Failed to save frame to {frame_path}")

        cap.release()
        return extracted_count

    def analyze_video_quality(self, video_path: str) -> Dict:
        """
        Analyze video to provide quality metrics and recommendations.

        Parameters:
        - video_path: Path to video file

        Returns:
        - Dictionary with analysis results
        """
        logger.info(f"Analyzing video quality: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sample_frames = np.linspace(0, total_frames - 1, min(50, total_frames), dtype=int)

        brightness_values = []
        contrast_values = []
        motion_values = []
        prev_gray = None

        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            brightness = float(np.mean(gray.astype(np.float32)))
            contrast = float(np.std(gray.astype(np.float32)))

            brightness_values.append(brightness)
            contrast_values.append(contrast)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion = float(np.mean(diff.astype(np.float32)))
                motion_values.append(motion)

            prev_gray = gray

        cap.release()

        try:
            dish_rois = detect_dishes_in_video(video_path, 0, save_detection_image=False)
            num_detected_dishes = len(dish_rois)
        except Exception as e:
            logger.warning(f"Failed to detect dishes: {e}")
            num_detected_dishes = 0

        analysis = {
            "video_info": {
                "total_frames": total_frames,
                "fps": fps,
                "duration_seconds": total_frames / fps if fps > 0 else 0,
                "resolution": f"{width}x{height}",
            },
            "quality_metrics": {
                "avg_brightness": np.mean(brightness_values),
                "brightness_std": np.std(brightness_values),
                "avg_contrast": np.mean(contrast_values),
                "contrast_std": np.std(contrast_values),
                "avg_motion": (np.mean(motion_values) if motion_values else 0),
            },
            "detection_results": {
                "num_dishes_detected": num_detected_dishes,
                "expected_dishes": (8 if self.use_detected_rois else len(config.ROI_BOXES)),
            },
            "recommendations": self._generate_recommendations(
                float(np.mean(brightness_values)),
                float(np.mean(contrast_values)),
                num_detected_dishes,
            ),
        }

        return analysis

    def _generate_recommendations(self, brightness: float, contrast: float, num_detected: int) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if brightness < 50:
            recommendations.append("Video appears too dark - consider increasing lighting")
        elif brightness > 200:
            recommendations.append("Video appears overexposed - consider reducing lighting")

        if contrast < 20:
            recommendations.append("Low contrast detected - may affect detection quality")

        if num_detected < 4:
            recommendations.append("Few dishes detected - check detection parameters " "or use manual ROIs")
        elif num_detected > 12:
            recommendations.append("Many dishes detected - may have false positives")

        if not recommendations:
            recommendations.append("Video quality appears suitable for processing")

        return recommendations


def extract_video_with_smart_detection(
    video_path: str,
    output_dir: str,
    num_frames: int = 225,
    dish_to_class: Optional[Dict[int, str]] = None,
    use_auto_detection: bool = True,
) -> int:
    """
    Convenience function for smart video extraction.

    Parameters:
    - video_path: Path to video file
    - output_dir: Output directory for frames
    - num_frames: Number of frames to extract per dish
    - dish_to_class: Mapping from dish index to class name
    - use_auto_detection: Whether to use automatic detection

    Returns:
    - Number of sequences created
    """
    extractor = SmartVideoExtractor(use_detected_rois=use_auto_detection)
    return extractor.extract_video_intelligently(video_path, output_dir, num_frames, dish_to_class)
