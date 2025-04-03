import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def read_training_log(log_file):
    """
    Reads the training_log.txt file with format:
      epoch,loss,accuracy
    Returns three lists: epochs, losses, accuracies.
    """
    epochs = []
    losses = []
    accuracies = []
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f) 
        for row in reader:
            # Convert to appropriate types
            epoch = int(row['Epoch'])
            loss = float(row['Loss'])
            acc = float(row['Accuracy'])
            
            epochs.append(epoch)
            losses.append(loss)
            accuracies.append(acc)
    
    return epochs, losses, accuracies

def plot_and_save_training_curve(epochs, losses, accuracies, output_file="training_curve.png"):
    """
    Generates and saves a plot showing the evolution of loss and accuracy.
    """
    plt.figure(figsize=(12, 6))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o', label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolution of loss")
    plt.legend()
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, marker='o', color="orange", label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Evolution of accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file) 
    #plt.show()
    print(f"Training plot saved to: {output_file}")

def read_final_metrics(metrics_file):
    """
    Reads the final_metrics.txt file with columns.
    Returns a dictionary with the class and its metrics.
    """
    metrics_dict = {}
    with open(metrics_file, 'r') as f:
        header = f.readline().strip()  
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            class_id = int(parts[0])
            precision = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            
            metrics_dict[class_id] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    return metrics_dict

def plot_and_save_final_metrics(metrics_dict, output_file="final_metrics.png"):
    """
    Generates a bar chart with precision, recall, and F1-Score per class,
    and saves it as a PNG file.
    """
    classes = sorted(metrics_dict.keys())
    precision_vals = [metrics_dict[c]['precision'] for c in classes]
    recall_vals = [metrics_dict[c]['recall'] for c in classes]
    f1_vals = [metrics_dict[c]['f1'] for c in classes]
    
    x = np.arange(len(classes))  # Positions for the classes
    width = 0.25  
    
    plt.figure(figsize=(12, 6))
    
    # Precision
    plt.bar(x - width, precision_vals, width, label='Precision', color='blue')
    # Recall
    plt.bar(x, recall_vals, width, label='Recall', color='green')
    # F1-Score
    plt.bar(x + width, f1_vals, width, label='F1-Score', color='orange')
    
    plt.xlabel("Class")
    plt.ylabel("Value")
    plt.title("Final metrics per class")
    plt.xticks(x, classes)  
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_file)
    #plt.show()
    print(f"Final metrics plot saved to: {output_file}")

if __name__ == "__main__":
    training_log_path = "results/training_log.txt"
    final_metrics_path = "results/final_metrics.txt"
    confusion_matrix_img = "C:results/confusion_matrix.png"
    
    if os.path.exists(training_log_path):
        epochs, losses, accuracies = read_training_log(training_log_path)
        plot_and_save_training_curve(epochs, losses, accuracies, output_file="results/training_curve.png")
    else:
        print(f"{training_log_path} not found. Training plot will not be generated.")
    
    if os.path.exists(final_metrics_path):
        metrics_dict = read_final_metrics(final_metrics_path)
        print("\n===== Final metrics per class =====")
        for c in sorted(metrics_dict.keys()):
            m = metrics_dict[c]
            print(f"Class {c}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")
        plot_and_save_final_metrics(metrics_dict, output_file="results/final_metrics.png")
    else:
        print(f"{final_metrics_path} not found. Final metrics plot will not be generated.")
