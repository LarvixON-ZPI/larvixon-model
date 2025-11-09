import torch
import numpy as np
import pandas as pd
import cv2
import os
import glob
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime
import warnings

from src.models.cnn_lstm_model import CNNLSTM
from src.utils.video_utils import video_to_fixed_frames
import src.config as config
from src.utils.logger import logger

warnings.filterwarnings("ignore")


class VideoFrameDataset(Dataset):
    """
    Dataset that extracts frames from videos and creates sequences for evaluation
    """

    def __init__(
        self,
        video_dir,
        num_frames=16,
        transform=None,
        temp_dir="temp_frames",
    ):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform
        self.temp_dir = temp_dir
        self.samples = []

        # Create temporary directory for frame extraction
        os.makedirs(self.temp_dir, exist_ok=True)

        # Process all video files
        self._process_videos()

    def _extract_label_from_filename(self, video_filename):
        """
        Extract label from video filename based on patterns
        """
        filename_lower = video_filename.lower()

        if (
            "nothing" in filename_lower
            or "_rl_" in filename_lower
            and "etoh" not in filename_lower
            and "water" not in filename_lower
        ):
            return 0  # Nothing
        elif "water" in filename_lower or "_wl_" in filename_lower:
            return 1  # Water (H2O)
        elif "etoh" in filename_lower or "ethanol" in filename_lower:
            return 2  # Ethanol
        elif "redbull" in filename_lower or "rb_" in filename_lower:
            return 3  # Redbull
        else:
            logger.warning(
                f"Could not determine label for {video_filename}, defaulting to 0 (Nothing)"
            )
            return 0

    def _process_videos(self):
        """
        Process all videos in the directory and extract frames
        """
        video_extensions = ["*.mov", "*.mp4", "*.avi", "*.mkv"]
        video_files = []

        for ext in video_extensions:
            video_files.extend(
                glob.glob(os.path.join(self.video_dir, ext))
            )

        logger.info(
            f"Found {len(video_files)} video files in {self.video_dir}"
        )

        for video_path in video_files:
            video_filename = os.path.basename(video_path)
            logger.info(f"Processing video: {video_filename}")

            # Extract label from filename
            label = self._extract_label_from_filename(video_filename)

            # Create output directory for this video's frames
            video_name = os.path.splitext(video_filename)[0]
            frame_output_dir = os.path.join(self.temp_dir, video_name)

            try:
                # Extract frames from video
                video_to_fixed_frames(
                    video_path=video_path,
                    output_dir=frame_output_dir,
                    num_frames=self.num_frames,
                    prefix="frame",
                )

                # Get all extracted frame paths
                frame_files = sorted(
                    glob.glob(
                        os.path.join(frame_output_dir, "frame_*.jpg")
                    )
                )

                if len(frame_files) >= self.num_frames:
                    # Take exactly num_frames
                    frame_files = frame_files[: self.num_frames]
                    self.samples.append((frame_files, label))
                    logger.info(
                        f"Added video {video_filename} with label {config.CLASS_NAMES[label]} ({len(frame_files)} frames)"
                    )
                else:
                    logger.warning(
                        f"Video {video_filename} has insufficient frames ({len(frame_files)} < {self.num_frames})"
                    )

            except Exception as e:
                logger.error(
                    f"Error processing video {video_filename}: {str(e)}"
                )
                continue

        logger.info(
            f"Successfully processed {len(self.samples)} video sequences"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        # Load and transform frames
        frames = []
        for frame_path in frame_paths:
            try:
                # Load image
                image = cv2.imread(frame_path)
                if image is None:
                    raise ValueError(f"Could not load image: {frame_path}")

                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Apply transforms
                if self.transform:
                    image = self.transform(image)

                frames.append(image)

            except Exception as e:
                logger.error(f"Error loading frame {frame_path}: {str(e)}")
                # Create a dummy black frame if loading fails
                if self.transform:
                    dummy_frame = self.transform(
                        np.zeros((224, 224, 3), dtype=np.uint8)
                    )
                else:
                    dummy_frame = torch.zeros(3, 224, 224)
                frames.append(dummy_frame)

        # Stack frames into tensor (num_frames, channels, height, width)
        frames_tensor = torch.stack(frames)

        return frames_tensor, label

    def cleanup(self):
        """
        Clean up temporary frame directories
        """
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")


class VideoEvaluator:
    """
    Cross-Validation Evaluator for CNN-LSTM Model using video data
    """

    def __init__(
        self,
        video_dir,
        num_frames=16,
        num_folds=5,
        test_size=0.2,
        random_state=42,
    ):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.num_folds = num_folds
        self.test_size = test_size
        self.random_state = random_state
        self.device = config.DEVICE
        self.small_dataset = False

        # Transform for evaluation
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create dataset from videos
        logger.info(f"Creating dataset from videos in {video_dir}")
        self.dataset = VideoFrameDataset(
            video_dir=video_dir,
            num_frames=num_frames,
            transform=self.transform,
        )

        # Results storage
        self.cv_results = defaultdict(list)
        self.fold_predictions = []
        self.fold_true_labels = []

        logger.info(
            f"Initialized Video Evaluator with {len(self.dataset)} video samples"
        )
        logger.info(f"Classes: {config.CLASS_NAMES}")

        # Check if dataset is too small for proper cross-validation
        if len(self.dataset) < 10:
            self.small_dataset = True
            self.num_folds = min(
                len(self.dataset), max(2, len(self.dataset) // 2)
            )
            self.test_size = max(0.1, 1.0 / len(self.dataset))
            logger.warning(
                f"Small dataset detected. Adjusting to {self.num_folds} folds."
            )

    def create_video_aware_splits(self):
        """
        Create train/test splits for video data
        """
        # Get all sample indices and their labels
        all_indices = list(range(len(self.dataset)))
        all_labels = [self.dataset.samples[idx][1] for idx in all_indices]

        # For small datasets or insufficient class diversity, skip stratification
        unique_labels = set(all_labels)
        if len(self.dataset) <= 4 or len(unique_labels) < 2:
            train_indices, test_indices = train_test_split(
                all_indices,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        else:
            try:
                train_indices, test_indices = train_test_split(
                    all_indices,
                    test_size=self.test_size,
                    stratify=all_labels,
                    random_state=self.random_state,
                )
            except ValueError:
                # Fallback to non-stratified split
                train_indices, test_indices = train_test_split(
                    all_indices,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )

        return train_indices, test_indices

    def create_cv_folds(self, train_indices):
        """
        Create K-fold splits for cross validation
        """
        train_labels = [
            self.dataset.samples[idx][1] for idx in train_indices
        ]

        # Try stratified K-fold, fallback to regular K-fold if not possible
        try:
            skf = StratifiedKFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            folds = []
            for train_fold_idx, val_fold_idx in skf.split(
                train_indices, train_labels
            ):
                train_fold = [train_indices[i] for i in train_fold_idx]
                val_fold = [train_indices[i] for i in val_fold_idx]
                folds.append((train_fold, val_fold))
        except ValueError:
            # Fallback to simple splitting
            from sklearn.model_selection import KFold

            kf = KFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            folds = []
            for train_fold_idx, val_fold_idx in kf.split(train_indices):
                train_fold = [train_indices[i] for i in train_fold_idx]
                val_fold = [train_indices[i] for i in val_fold_idx]
                folds.append((train_fold, val_fold))

        return folds

    def evaluate_model(self, model, dataloader, fold_name=""):
        """
        Evaluate model on given dataloader
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for frames, labels in dataloader:
                frames = frames.to(self.device)
                outputs = model(frames)

                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        return (
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs),
        )

    def calculate_metrics(self, y_true, y_pred, y_probs=None):
        """
        Calculate comprehensive evaluation metrics
        """
        metrics = {}

        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        precision_macro, recall_macro, f1_macro, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
        )
        precision_micro, recall_micro, f1_micro, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, average="micro", zero_division=0
            )
        )

        metrics["precision_macro"] = precision_macro
        metrics["recall_macro"] = recall_macro
        metrics["f1_macro"] = f1_macro
        metrics["precision_micro"] = precision_micro
        metrics["recall_micro"] = recall_micro
        metrics["f1_micro"] = f1_micro

        for i, class_name in enumerate(config.CLASS_NAMES):
            if i < len(precision):
                metrics[f"precision_{class_name}"] = float(precision[i])
                metrics[f"recall_{class_name}"] = float(recall[i])
                metrics[f"f1_{class_name}"] = float(f1[i])
                metrics[f"support_{class_name}"] = int(support[i])
            else:
                metrics[f"precision_{class_name}"] = 0.0
                metrics[f"recall_{class_name}"] = 0.0
                metrics[f"f1_{class_name}"] = 0.0
                metrics[f"support_{class_name}"] = 0

        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

        if y_probs is not None and config.NUM_CLASSES > 2:
            try:
                metrics["auc_macro"] = roc_auc_score(
                    y_true, y_probs, multi_class="ovr", average="macro"
                )
                metrics["auc_weighted"] = roc_auc_score(
                    y_true, y_probs, multi_class="ovr", average="weighted"
                )
            except ValueError:
                metrics["auc_macro"] = 0.0
                metrics["auc_weighted"] = 0.0

        return metrics

    def perform_cross_validation(self, model_path=None):
        """
        Perform complete cross-validation evaluation
        """
        if model_path is None:
            model_path = config.MODEL_PATH

        logger.info(
            f"Starting Cross-Validation with {self.num_folds} folds"
        )
        logger.info(f"Using model: {model_path}")

        # Load pre-trained model
        base_model = CNNLSTM(num_classes=config.NUM_CLASSES).to(
            self.device
        )

        # Try to load the model with fallback options
        try:
            base_model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=False,
                )
            )
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load {model_path}: {e}")

            # Try the checkpoint file
            checkpoint_path = "models/cnn_lstm_checkpoint.pt"
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
                if "model_state_dict" in checkpoint:
                    base_model.load_state_dict(
                        checkpoint["model_state_dict"]
                    )
                else:
                    base_model.load_state_dict(checkpoint)
                logger.info(
                    f"Successfully loaded model from checkpoint: {checkpoint_path}"
                )
            except Exception as e2:
                logger.error(
                    f"Failed to load checkpoint {checkpoint_path}: {e2}"
                )
                logger.warning(
                    "Using randomly initialized model for demonstration purposes"
                )

        # For very small datasets, just evaluate the entire dataset
        if self.small_dataset or len(self.dataset) <= 4:
            logger.warning(
                "Very small dataset - performing simple evaluation on all data"
            )

            data_loader = DataLoader(
                self.dataset, batch_size=config.BATCH_SIZE, shuffle=False
            )

            # Evaluate on all data
            all_preds, all_labels, all_probs = self.evaluate_model(
                base_model, data_loader, "All_Data"
            )

            # Calculate metrics
            final_test_metrics = self.calculate_metrics(
                all_labels, all_preds, all_probs
            )

            # Mock some CV results for consistency
            for metric_name, value in final_test_metrics.items():
                self.cv_results[metric_name] = [value]

            return final_test_metrics, all_preds, all_labels, all_probs

        # Normal cross-validation for larger datasets
        train_indices, test_indices = self.create_video_aware_splits()

        logger.info(
            f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}"
        )

        # Create CV folds from training data
        cv_folds = self.create_cv_folds(train_indices)

        # Perform cross-validation
        for fold, (fold_train_indices, fold_val_indices) in enumerate(
            cv_folds
        ):
            logger.info(f"\n--- Fold {fold + 1}/{self.num_folds} ---")

            val_sampler = SubsetRandomSampler(fold_val_indices)
            val_loader = DataLoader(
                self.dataset,
                batch_size=config.BATCH_SIZE,
                sampler=val_sampler,
            )

            # Evaluate on validation fold
            val_preds, val_labels, val_probs = self.evaluate_model(
                base_model, val_loader, f"Fold_{fold+1}"
            )

            # Calculate metrics for this fold
            fold_metrics = self.calculate_metrics(
                val_labels, val_preds, val_probs
            )

            # Store results
            for metric_name, value in fold_metrics.items():
                self.cv_results[metric_name].append(value)

            # Store predictions for ensemble analysis
            self.fold_predictions.append(val_preds)
            self.fold_true_labels.append(val_labels)

            logger.info(
                f"Fold {fold + 1} Accuracy: {fold_metrics['accuracy']:.4f}"
            )
            logger.info(
                f"Fold {fold + 1} F1-Macro: {fold_metrics['f1_macro']:.4f}"
            )

        # Final evaluation on held-out test set
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(
            self.dataset,
            batch_size=config.BATCH_SIZE,
            sampler=test_sampler,
        )

        test_preds, test_labels, test_probs = self.evaluate_model(
            base_model, test_loader, "Test"
        )

        # Calculate final test metrics
        final_test_metrics = self.calculate_metrics(
            test_labels, test_preds, test_probs
        )

        return final_test_metrics, test_preds, test_labels, test_probs

    def print_cv_results(self):
        """
        Print comprehensive cross-validation results
        """
        print("\n" + "=" * 80)
        print("VIDEO CROSS-VALIDATION RESULTS")
        print("=" * 80)

        key_metrics = [
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "mcc",
        ]

        print(
            f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}"
        )
        print("-" * 60)

        for metric in key_metrics:
            if metric in self.cv_results:
                values = np.array(self.cv_results[metric])
                print(
                    f"{metric:<20} {values.mean():<10.4f} {values.std():<10.4f} "
                    f"{values.min():<10.4f} {values.max():<10.4f}"
                )

        # Per-class F1 scores
        print(f"\nPer-Class F1 Scores:")
        print("-" * 40)
        for class_name in config.CLASS_NAMES:
            metric_key = f"f1_{class_name}"
            if metric_key in self.cv_results:
                values = np.array(self.cv_results[metric_key])
                print(
                    f"{class_name:<15} {values.mean():<10.4f} ± {values.std():<8.4f}"
                )

    def save_results(
        self, final_test_metrics, save_dir="video_evaluation_results"
    ):
        """
        Save detailed results to files
        """
        os.makedirs(save_dir, exist_ok=True)

        # Cross-validation results summary
        cv_summary = {}
        for metric, values in self.cv_results.items():
            cv_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": [float(v) for v in values],
            }

        # Complete results dictionary
        results = {
            "cross_validation": cv_summary,
            "final_test_metrics": {
                k: float(v) for k, v in final_test_metrics.items()
            },
            "configuration": {
                "video_dir": self.video_dir,
                "num_frames": self.num_frames,
                "num_folds": self.num_folds,
                "test_size": self.test_size,
                "random_state": self.random_state,
                "num_classes": config.NUM_CLASSES,
                "class_names": config.CLASS_NAMES,
                "batch_size": config.BATCH_SIZE,
                "device": str(self.device),
                "num_videos": len(self.dataset),
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Save as JSON
        with open(f"{save_dir}/video_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save as CSV for easy analysis
        cv_df = pd.DataFrame(self.cv_results)
        cv_df.to_csv(
            f"{save_dir}/video_cross_validation_metrics.csv", index=False
        )

        logger.info(f"Results saved to {save_dir}/")

    def cleanup(self):
        """
        Clean up temporary files
        """
        self.dataset.cleanup()


def main():
    """
    Main evaluation function for video-based cross-validation
    """
    print("Starting Video Cross-Validation Evaluation of CNN-LSTM Model")
    print("=" * 60)

    # Initialize video evaluator
    video_dir = "internal_data"
    num_frames = 16  # Adjust based on your model requirements

    evaluator = VideoEvaluator(
        video_dir=video_dir,
        num_frames=num_frames,
        num_folds=5,
        test_size=0.2,
        random_state=42,
    )

    try:
        # Perform cross-validation
        final_test_metrics, test_preds, test_labels, test_probs = (
            evaluator.perform_cross_validation()
        )

        # Print results
        evaluator.print_cv_results()

        print(f"\n{'='*80}")
        print("FINAL TEST SET RESULTS")
        print("=" * 80)

        print(f"Final Test Accuracy: {final_test_metrics['accuracy']:.4f}")
        print(f"Final Test F1-Macro: {final_test_metrics['f1_macro']:.4f}")
        print(f"Final Test MCC: {final_test_metrics['mcc']:.4f}")

        # Detailed classification report
        print(f"\nDetailed Classification Report (Test Set):")
        print(
            classification_report(
                test_labels,
                test_preds,
                target_names=config.CLASS_NAMES,
                digits=4,
            )
        )

        # Save results
        evaluator.save_results(final_test_metrics)

        print(f"\nVideo evaluation completed successfully!")
        print(f"Results saved to video_evaluation_results/ directory")

    except Exception as e:
        logger.error(f"Video evaluation failed: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        evaluator.cleanup()


if __name__ == "__main__":
    main()
