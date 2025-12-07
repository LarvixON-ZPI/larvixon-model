import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler
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
import os
import json
from datetime import datetime
import warnings

from src.datasets.frame_dataset import FrameDataset
from src.models.cnn_lstm_model import CNNLSTM
import src.config as config
from src.utils.logger import logger

warnings.filterwarnings("ignore")


class CrossValidationEvaluator:
    """
    Comprehensive Cross-Validation Evaluator for CNN-LSTM Model
    """

    def __init__(self, data_dir, num_folds=5, test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.num_folds = num_folds
        self.test_size = test_size
        self.random_state = random_state
        self.device = config.DEVICE
        self.small_dataset = False

        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        temp_dataset = FrameDataset(root_dir=self.data_dir, num_frames=16, transform=self.transform)

        if len(temp_dataset) == 0:
            for frames_count in [10, 8, 5]:
                temp_dataset = FrameDataset(
                    root_dir=self.data_dir,
                    num_frames=frames_count,
                    transform=self.transform,
                )
                if len(temp_dataset) > 0:
                    logger.info(f"Using {frames_count} frames per sequence")
                    break

        self.dataset = temp_dataset
        self.actual_num_frames = self.dataset.num_frames if hasattr(self.dataset, "num_frames") else 16

        self.cv_results = defaultdict(list)
        self.fold_predictions = []
        self.fold_true_labels = []

        logger.info(f"Initialized CV Evaluator with {len(self.dataset)} samples")
        logger.info(f"Classes: {config.CLASS_NAMES}")

        if len(self.dataset) < 10:
            self.small_dataset = True
            self.num_folds = min(len(self.dataset), 4)
            self.test_size = 1.0 / len(self.dataset)
            logger.warning(f"Small dataset detected. Adjusting to {self.num_folds} folds.")

    def create_sequence_aware_splits(self):
        """
        Create train/test splits that respect sequence boundaries to prevent data leakage
        """
        sequence_groups = defaultdict(list)
        sequence_labels = []

        for idx, (frame_paths, label) in enumerate(self.dataset.samples):
            seq_dir = os.path.dirname(frame_paths[0])
            sequence_groups[seq_dir].append(idx)
            if seq_dir not in [item[0] for item in sequence_labels]:
                sequence_labels.append((seq_dir, label))

        seq_dirs = [item[0] for item in sequence_labels]
        labels = [item[1] for item in sequence_labels]

        if len(seq_dirs) <= 4 or len(set(labels)) < 2:
            train_seqs, test_seqs, train_labels, test_labels = train_test_split(
                seq_dirs,
                labels,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        else:
            train_seqs, test_seqs, train_labels, test_labels = train_test_split(
                seq_dirs,
                labels,
                test_size=self.test_size,
                stratify=labels,
                random_state=self.random_state,
            )

        train_indices = []
        test_indices = []

        for seq_dir in train_seqs:
            train_indices.extend(sequence_groups[seq_dir])

        for seq_dir in test_seqs:
            test_indices.extend(sequence_groups[seq_dir])

        return train_indices, test_indices, train_seqs, test_seqs

    def create_cv_folds(self, train_indices):
        """
        Create stratified K-fold splits for cross validation
        """
        train_labels = [self.dataset.samples[idx][1] for idx in train_indices]

        skf = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        folds = []
        for train_fold_idx, val_fold_idx in skf.split(train_indices, train_labels):
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

        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
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
                metrics["auc_macro"] = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
                metrics["auc_weighted"] = roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")
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

        logger.info(f"Starting Cross-Validation with {self.num_folds} folds")
        logger.info(f"Using model: {model_path}")

        base_model = CNNLSTM(num_classes=config.NUM_CLASSES).to(self.device)

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

            checkpoint_path = "models/cnn_lstm_checkpoint.pt"
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
                if "model_state_dict" in checkpoint:
                    base_model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    base_model.load_state_dict(checkpoint)
                logger.info(f"Successfully loaded model from checkpoint: {checkpoint_path}")
            except Exception as e2:
                logger.error(f"Failed to load checkpoint {checkpoint_path}: {e2}")
                logger.warning("Using randomly initialized model for demonstration purposes")

        if self.small_dataset or len(self.dataset) <= 8:
            logger.warning("Very small dataset - performing simple evaluation on all data")

            all_indices = list(range(len(self.dataset)))
            data_loader = DataLoader(self.dataset, batch_size=config.BATCH_SIZE, shuffle=False)

            all_preds, all_labels, all_probs = self.evaluate_model(base_model, data_loader, "All_Data")

            final_test_metrics = self.calculate_metrics(all_labels, all_preds, all_probs)

            for metric_name, value in final_test_metrics.items():
                self.cv_results[metric_name] = [value]  # Single "fold"

            return final_test_metrics, all_preds, all_labels, all_probs

        train_indices, test_indices, train_seqs, test_seqs = self.create_sequence_aware_splits()

        logger.info(f"Train sequences: {len(train_seqs)}, Test sequences: {len(test_seqs)}")
        logger.info(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")

        cv_folds = self.create_cv_folds(train_indices)

        for fold, (fold_train_indices, fold_val_indices) in enumerate(cv_folds):
            logger.info(f"\n--- Fold {fold + 1}/{self.num_folds} ---")

            val_sampler = SubsetRandomSampler(fold_val_indices)
            val_loader = DataLoader(
                self.dataset,
                batch_size=config.BATCH_SIZE,
                sampler=val_sampler,
            )

            val_preds, val_labels, val_probs = self.evaluate_model(base_model, val_loader, f"Fold_{fold+1}")

            fold_metrics = self.calculate_metrics(val_labels, val_preds, val_probs)

            for metric_name, value in fold_metrics.items():
                self.cv_results[metric_name].append(value)

            self.fold_predictions.append(val_preds)
            self.fold_true_labels.append(val_labels)

            logger.info(f"Fold {fold + 1} Accuracy: {fold_metrics['accuracy']:.4f}")
            logger.info(f"Fold {fold + 1} F1-Macro: {fold_metrics['f1_macro']:.4f}")

        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(
            self.dataset,
            batch_size=config.BATCH_SIZE,
            sampler=test_sampler,
        )

        test_preds, test_labels, test_probs = self.evaluate_model(base_model, test_loader, "Test")

        final_test_metrics = self.calculate_metrics(test_labels, test_preds, test_probs)

        return final_test_metrics, test_preds, test_labels, test_probs

    def print_cv_results(self):
        """
        Print comprehensive cross-validation results
        """
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 80)

        key_metrics = [
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "mcc",
        ]

        print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 60)

        for metric in key_metrics:
            if metric in self.cv_results:
                values = np.array(self.cv_results[metric])
                print(
                    f"{metric:<20} {values.mean():<10.4f} {values.std():<10.4f} " f"{values.min():<10.4f} {values.max():<10.4f}"
                )

        print(f"\nPer-Class F1 Scores:")
        print("-" * 40)
        for class_name in config.CLASS_NAMES:
            metric_key = f"f1_{class_name}"
            if metric_key in self.cv_results:
                values = np.array(self.cv_results[metric_key])
                print(f"{class_name:<15} {values.mean():<10.4f} ± {values.std():<8.4f}")

    def plot_results(self, test_preds, test_labels, save_dir="evaluation_results"):
        """
        Create visualization plots
        """
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(15, 10))

        key_metrics = [
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
        ]

        plt.subplot(2, 3, 1)
        metrics_data = [self.cv_results[metric] for metric in key_metrics if metric in self.cv_results]
        metrics_labels = [metric for metric in key_metrics if metric in self.cv_results]

        if metrics_data:
            plt.boxplot(metrics_data, labels=metrics_labels)
            plt.title("Cross-Validation Metrics Distribution")
            plt.xticks(rotation=45)
            plt.ylabel("Score")

        plt.subplot(2, 3, 2)
        class_f1_means = []
        class_f1_stds = []
        class_names_plot = []

        for class_name in config.CLASS_NAMES:
            metric_key = f"f1_{class_name}"
            if metric_key in self.cv_results:
                values = np.array(self.cv_results[metric_key])
                class_f1_means.append(values.mean())
                class_f1_stds.append(values.std())
                class_names_plot.append(class_name)

        if class_f1_means:
            x_pos = np.arange(len(class_names_plot))
            plt.bar(
                x_pos,
                class_f1_means,
                yerr=class_f1_stds,
                alpha=0.7,
                capsize=5,
            )
            plt.xlabel("Classes")
            plt.ylabel("F1 Score")
            plt.title("Per-Class F1 Scores (Mean ± Std)")
            plt.xticks(x_pos, class_names_plot, rotation=45)

        plt.subplot(2, 3, 3)
        cm = confusion_matrix(test_labels, test_preds)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES,
        )
        plt.title("Test Set Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.subplot(2, 3, 4)
        fold_numbers = list(range(1, len(self.cv_results["accuracy"]) + 1))
        plt.plot(
            fold_numbers,
            self.cv_results["accuracy"],
            "bo-",
            linewidth=2,
            markersize=8,
        )
        plt.axhline(
            y=np.mean(self.cv_results["accuracy"]),
            color="r",
            linestyle="--",
            label=f'Mean: {np.mean(self.cv_results["accuracy"]):.4f}',
        )
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Across CV Folds")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 5)
        metrics_std = {metric: np.std(values) for metric, values in self.cv_results.items() if metric in key_metrics}
        if metrics_std:
            plt.bar(metrics_std.keys(), metrics_std.values(), alpha=0.7)
            plt.title("Metric Stability (Lower is Better)")
            plt.ylabel("Standard Deviation")
            plt.xticks(rotation=45)

        plt.subplot(2, 3, 6)
        unique_labels, counts = np.unique(test_labels, return_counts=True)
        class_names_dist = [config.CLASS_NAMES[i] for i in unique_labels]
        plt.pie(
            counts,
            labels=class_names_dist,
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title("Test Set Class Distribution")

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/cross_validation_results.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        plt.figure(figsize=(10, 8))
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES,
        )
        plt.title("Normalized Confusion Matrix (Test Set)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/confusion_matrix_normalized.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def save_results(self, final_test_metrics, save_dir="evaluation_results"):
        """
        Save detailed results to files
        """
        os.makedirs(save_dir, exist_ok=True)

        cv_summary = {}
        for metric, values in self.cv_results.items():
            cv_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": [float(v) for v in values],
            }

        results = {
            "cross_validation": cv_summary,
            "final_test_metrics": {k: float(v) for k, v in final_test_metrics.items()},
            "configuration": {
                "num_folds": self.num_folds,
                "test_size": self.test_size,
                "random_state": self.random_state,
                "num_classes": config.NUM_CLASSES,
                "class_names": config.CLASS_NAMES,
                "num_frames": config.NUM_FRAMES,
                "batch_size": config.BATCH_SIZE,
                "device": str(self.device),
            },
            "timestamp": datetime.now().isoformat(),
        }

        with open(f"{save_dir}/evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        cv_df = pd.DataFrame(self.cv_results)
        cv_df.to_csv(f"{save_dir}/cross_validation_metrics.csv", index=False)

        logger.info(f"Results saved to {save_dir}/")


def main():
    """
    Main evaluation function
    """
    print("Starting Cross-Validation Evaluation of CNN-LSTM Model")
    print("=" * 60)

    evaluator = CrossValidationEvaluator(
        data_dir=config.DATA_DIR,
        num_folds=5,
        test_size=0.2,
        random_state=42,
    )

    try:
        final_test_metrics, test_preds, test_labels, test_probs = evaluator.perform_cross_validation()

        evaluator.print_cv_results()

        print(f"\n{'='*80}")
        print("FINAL TEST SET RESULTS")
        print("=" * 80)

        print(f"Final Test Accuracy: {final_test_metrics['accuracy']:.4f}")
        print(f"Final Test F1-Macro: {final_test_metrics['f1_macro']:.4f}")
        print(f"Final Test MCC: {final_test_metrics['mcc']:.4f}")

        print(f"\nDetailed Classification Report (Test Set):")
        print(
            classification_report(
                test_labels,
                test_preds,
                target_names=config.CLASS_NAMES,
                digits=4,
            )
        )

        evaluator.plot_results(test_preds, test_labels)

        evaluator.save_results(final_test_metrics)

        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to evaluation_results/ directory")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
