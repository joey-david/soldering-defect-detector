import time
import torch
from api import initialize_model
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def test_dataset(model, engine, dataset_path, results_dir="results"):
    # Collect stats
    all_times = []
    all_labels = []
    all_preds = []
    anomaly_map_sum = None
    count = 0
    # count all images in the folder and its subfolders
    total_images = sum(1 for _ in Path(dataset_path).rglob("*.png"))

    for image_path in Path(dataset_path).rglob("*.png"):
        start = time.time()
        batch_preds = engine.predict(
            model=model,
            dataset=PredictDataset(path=image_path)
        )
        elapsed = time.time() - start
        all_times.append(elapsed)

        # Derive label from folder structure
        # 0 = Sans_Defaut, else = Defaut
        label = 0 if "Sans_Defaut" in str(image_path) else 1
        if not batch_preds:
            continue

        prediction = batch_preds[0]
        pred_label = prediction["pred_labels"].item()
        all_labels.append(label)
        all_preds.append(pred_label)

        # Accumulate anomaly map for averaging
        if "anomaly_maps" in prediction:
            anomaly_map = prediction["anomaly_maps"].squeeze().cpu().numpy()
        elif "distance" in prediction:
            anomaly_map = prediction["distance"].squeeze().cpu().numpy()
        else:
            continue

        if anomaly_map_sum is None:
            anomaly_map_sum = anomaly_map
        else:
            anomaly_map_sum += anomaly_map
        count += 1
        print(f"{round(100*count / total_images,2)}% done")

    # Compute metrics
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("Confusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Sans_Defaut","Defaut"], 
                yticklabels=["Sans_Defaut","Defaut"])
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Plot average anomaly map
    if anomaly_map_sum is not None and count > 0:
        avg_map = anomaly_map_sum / count
        plt.figure(figsize=(4,4))
        plt.imshow(avg_map, cmap='jet')
        plt.title("Average Anomaly Map")
        plt.axis('off')
        plt.savefig("average_anomaly_map.png")
        plt.close()

    # Plot inference times
    plt.figure(figsize=(6,4))
    plt.plot(all_times)
    plt.title("Inference Time per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Time (seconds)")
    plt.savefig("inference_time.png")
    plt.close()

def main():
    binary_model, binary_engine = initialize_model('./models/model_bin.pth')
    dataset_path = "data/dataset"
    test_dataset(binary_model, binary_engine, dataset_path)

if __name__ == "__main__":
    main()