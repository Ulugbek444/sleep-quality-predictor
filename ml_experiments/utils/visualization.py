import numpy as np
import os
import matplotlib.pyplot as plt
import mlflow
from sklearn.preprocessing import label_binarize
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay, RocCurveDisplay
from ml_experiments.config.experiment_config import (
    BASE_DIR, REPORTS_DIR,
    FIGURES_DIR, ROC_DIR, PRECISION_RECALL_DIR, CONF_MATRIX_DIR
)


def save_precision_recall_curve(y_true, y_prob, run_name, dataset_type):
    """Сохраняет Precision-Recall кривые для каждого класса"""
    try:
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)

        for i, class_label in enumerate(classes):
            PrecisionRecallDisplay.from_predictions(y_true_bin[:, i], y_prob[:, i])
            plt.title(f"Precision-Recall Curve: {run_name} ({dataset_type}) - class {class_label}")
            pr_path = os.path.join(PRECISION_RECALL_DIR, f"{run_name}_{dataset_type}_class_{class_label}.png")
            plt.savefig(pr_path, dpi=100, bbox_inches='tight')
            mlflow.log_artifact(pr_path)
            plt.close()
    except Exception as e:
        print(f"Предупреждение: не удалось сохранить Precision-Recall кривые: {e}")


def save_roc_curve(y_true, y_prob, run_name, dataset_type):
    """Сохраняет ROC кривые для каждого класса"""
    try:
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)

        for i, class_label in enumerate(classes):
            RocCurveDisplay.from_predictions(y_true_bin[:, i], y_prob[:, i])
            plt.title(f"ROC Curve: {run_name} ({dataset_type}) - class {class_label}")
            roc_path = os.path.join(ROC_DIR, f"{run_name}_{dataset_type}_class_{class_label}.png")
            plt.savefig(roc_path, dpi=100, bbox_inches='tight')
            mlflow.log_artifact(roc_path)
            plt.close()
    except Exception as e:
        print(f"Предупреждение: не удалось сохранить ROC кривые: {e}")


def save_confusion_matrix(y_true, y_pred, run_name, dataset_type):
    """Сохраняет матрицу ошибок"""
    try:
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=["bad", "medium", "good"],
            cmap=plt.cm.Blues
        )
        plt.title(f"Confusion Matrix: {run_name} ({dataset_type})")
        cm_path = os.path.join(CONF_MATRIX_DIR, f"{run_name}_{dataset_type}.png")
        plt.savefig(cm_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        plt.close()
    except Exception as e:
        print(f"Предупреждение: не удалось сохранить матрицу ошибок: {e}")
