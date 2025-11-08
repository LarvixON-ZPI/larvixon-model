#!/usr/bin/env python3
"""
Validation script for the smart petri dish detection system.
Tests detection accuracy and provides quality metrics.
"""

import os
import sys
sys.path.append(".")
import cv2
import json
import argparse
from typing import List, Tuple, Dict
from src.utils.logger import logger
from src.utils.petri_dish_detector import detect_dishes_in_video
from src.utils.larvae_detector import LarvaeDetector
from src.utils.smart_extraction import SmartVideoExtractor
import src.config as config


def load_ground_truth_rois(file_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Load ground truth ROI boxes from JSON file.
 
    Parameters:
    - file_path: Path to JSON file with ground truth ROIs

    Returns:
    - List of ROI tuples (x, y, w, h)
    """
    if not os.path.exists(file_path):
        logger.warning(f"Ground truth file {file_path} not found")
        return []

    with open(file_path, 'r') as f:
        roi_data = json.load(f)

    return [tuple(roi) for roi in roi_data]


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1: First bounding box (x, y, w, h)
    - box2: Second bounding box (x, y, w, h)

    Returns:
    - IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def evaluate_detection_accuracy(detected_rois: List[Tuple[int, int, int, int]],
                              ground_truth_rois: List[Tuple[int, int, int, int]],
                              iou_threshold: float = 0.5) -> Dict:
    """
    Evaluate detection accuracy against ground truth.

    Parameters:
    - detected_rois: List of detected ROIs
    - ground_truth_rois: List of ground truth ROIs
    - iou_threshold: IoU threshold for considering a detection correct

    Returns:
    - Dictionary with evaluation metrics
    """
    if not ground_truth_rois:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "num_detected": len(detected_rois),
            "num_ground_truth": 0,
            "matches": []
        }

    matches = []
    gt_matched = [False] * len(ground_truth_rois)

    for i, detected_roi in enumerate(detected_rois):
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt_roi in enumerate(ground_truth_rois):
            if gt_matched[j]:
                continue

            iou = calculate_iou(detected_roi, gt_roi)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx != -1:
            matches.append({
                "detected_idx": i,
                "gt_idx": best_gt_idx,
                "iou": best_iou,
                "detected_roi": detected_roi,
                "gt_roi": ground_truth_rois[best_gt_idx]
            })
            gt_matched[best_gt_idx] = True

    true_positives = len(matches)
    false_positives = len(detected_rois) - true_positives
    false_negatives = len(ground_truth_rois) - true_positives

    precision = true_positives / len(detected_rois) if detected_rois else 0.0
    recall = true_positives / len(ground_truth_rois)
    f1_score = (2 * precision * recall / (precision + recall) 
               if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "num_detected": len(detected_rois),
        "num_ground_truth": len(ground_truth_rois),
        "matches": matches
    }


def test_single_video(video_path: str, 
                     ground_truth_file: str = "roi_boxes.json") -> Dict:
    """
    Test detection system on a single video.

    Parameters:
    - video_path: Path to video file
    - ground_truth_file: Path to ground truth ROI file

    Returns:
    - Dictionary with test results
    """
    logger.info(f"Testing detection on video: {video_path}")

    ground_truth_rois = load_ground_truth_rois(ground_truth_file)

    try:
        detected_rois = detect_dishes_in_video(video_path, 0, 
                                              save_detection_image=True)

        detection_metrics = evaluate_detection_accuracy(detected_rois, 
                                                       ground_truth_rois)

        larvae_results = {}
        if detected_rois:
            detector = LarvaeDetector()
            larvae_start_frames = detector.detect_larvae_appearance(
                video_path, detected_rois[:4], max_frames_check=200)  # Test first 4 dishes

            larvae_results = {
                "larvae_detected_dishes": sum(1 for frame in larvae_start_frames if frame > 0),
                "total_dishes_tested": len(larvae_start_frames),
                "start_frames": larvae_start_frames
            }

        extractor = SmartVideoExtractor()
        quality_analysis = extractor.analyze_video_quality(video_path)

        return {
            "video_path": video_path,
            "detection_metrics": detection_metrics,
            "larvae_results": larvae_results,
            "quality_analysis": quality_analysis,
            "success": True
        }

    except Exception as e:
        logger.error(f"Test failed for {video_path}: {e}")
        return {
            "video_path": video_path,
            "error": str(e),
            "success": False
        }


def run_validation_suite(test_video_dir: str = "test_videos/",
                        ground_truth_file: str = "roi_boxes.json") -> Dict:
    """
    Run validation tests on multiple videos.

    Parameters:
    - test_video_dir: Directory containing test videos
    - ground_truth_file: Path to ground truth ROI file

    Returns:
    - Dictionary with aggregated results
    """
    logger.info("Running validation suite for petri dish detection system")

    if not os.path.exists(test_video_dir):
        logger.warning(f"Test video directory {test_video_dir} not found")
        return {"error": "Test directory not found"}

    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    test_videos = []

    for file in os.listdir(test_video_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            test_videos.append(os.path.join(test_video_dir, file))

    if not test_videos:
        logger.warning("No test videos found")
        return {"error": "No test videos found"}

    logger.info(f"Found {len(test_videos)} test videos")

    results = []
    successful_tests = 0

    for video_path in test_videos:
        result = test_single_video(video_path, ground_truth_file)
        results.append(result)

        if result["success"]:
            successful_tests += 1
            metrics = result["detection_metrics"]
            logger.info(f"Video {os.path.basename(video_path)}: "
                       f"P={metrics['precision']:.3f}, "
                       f"R={metrics['recall']:.3f}, "
                       f"F1={metrics['f1_score']:.3f}")
        else:
            logger.error(f"Failed: {os.path.basename(video_path)}")

    successful_results = [r for r in results if r["success"]]

    if successful_results:
        avg_precision = sum(r["detection_metrics"]["precision"] 
                          for r in successful_results) / len(successful_results)
        avg_recall = sum(r["detection_metrics"]["recall"] 
                        for r in successful_results) / len(successful_results)
        avg_f1 = sum(r["detection_metrics"]["f1_score"] 
                    for r in successful_results) / len(successful_results)

        aggregate_stats = {
            "total_videos": len(test_videos),
            "successful_tests": successful_tests,
            "success_rate": successful_tests / len(test_videos),
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1_score": avg_f1
        }
    else:
        aggregate_stats = {
            "total_videos": len(test_videos),
            "successful_tests": 0,
            "success_rate": 0.0,
            "average_precision": 0.0,
            "average_recall": 0.0,
            "average_f1_score": 0.0
        }

    return {
        "aggregate_stats": aggregate_stats,
        "individual_results": results
    }


def create_validation_report(results: Dict, output_file: str = "validation_report.json"):
    """
    Create a detailed validation report.

    Parameters:
    - results: Results from validation suite
    - output_file: Output file for the report
    """
    logger.info(f"Creating validation report: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if "aggregate_stats" in results:
        stats = results["aggregate_stats"]
        print("\n" + "="*60)
        print("VALIDATION REPORT SUMMARY")
        print("="*60)
        print(f"Total videos tested: {stats['total_videos']}")
        print(f"Successful tests: {stats['successful_tests']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Average Precision: {stats['average_precision']:.3f}")
        print(f"Average Recall: {stats['average_recall']:.3f}")
        print(f"Average F1-Score: {stats['average_f1_score']:.3f}")
        print("="*60)

    logger.info(f"Validation report saved to {output_file}")


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate petri dish detection system")
    parser.add_argument("--video", type=str, help="Single video file to test")
    parser.add_argument("--test_dir", type=str, default="test_videos/",
                       help="Directory containing test videos")
    parser.add_argument("--ground_truth", type=str, default="roi_boxes.json",
                       help="Ground truth ROI file")
    parser.add_argument("--output", type=str, default="validation_report.json",
                       help="Output report file")

    args = parser.parse_args()

    if args.video:
        result = test_single_video(args.video, args.ground_truth)
        print(json.dumps(result, indent=2))
    else:
        results = run_validation_suite(args.test_dir, args.ground_truth)
        create_validation_report(results, args.output)


if __name__ == "__main__":
    main()
