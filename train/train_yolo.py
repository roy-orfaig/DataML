import mlflow
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Set MLflow tracking

mlflow.set_tracking_uri("http://localhost:5000")  # Change if using a remote MLflow server
mlflow.set_experiment("YOLOv8 Training with Full Metrics")

# mlflow.set_tracking_uri("file:///home/roy.o@uveye.local/mlruns")  # Change path as needed
# mlflow.set_experiment("YOLOv8 Training with Full Metrics")


# Define dataset and training parameters
data_yaml = "/home/roy.o@uveye.local/projects/Data/tile_1024/data.yaml"  # Update with actual path
epochs = 1
batch_size = 4
img_size = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Choose appropriate model size (n, s, m, l, x)

with mlflow.start_run():
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("image_size", img_size)
    mlflow.log_param("device", device)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        verbose=True
    )

    # Log final metrics after training
    metrics = results.metrics
    mlflow.log_metric("train_loss", metrics["loss"])
    mlflow.log_metric("val_loss", metrics["val/loss"])
    mlflow.log_metric("precision_mean", metrics["metrics/precision(B)"])
    mlflow.log_metric("recall_mean", metrics["metrics/recall(B)"])
    mlflow.log_metric("F1_score", metrics["metrics/F1(B)"])
    mlflow.log_metric("mAP50", metrics["metrics/mAP50(B)"])
    mlflow.log_metric("mAP50-95", metrics["metrics/mAP50-95(B)"])

    # Log per-class precision, recall, and F1-score
    for i, cls in enumerate(metrics["class/labels"]):
        mlflow.log_metric(f"precision_class_{cls}", metrics["metrics/precision"][i])
        mlflow.log_metric(f"recall_class_{cls}", metrics["metrics/recall"][i])
        mlflow.log_metric(f"F1_class_{cls}", metrics["metrics/F1"][i])
    
    # Log confusion matrix
    confusion_matrix = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Save trained model
    model_path = "runs/train/yolov8_custom"
    mlflow.log_artifact(model_path)

print("Training complete. Check MLflow dashboard for all logged metrics.")