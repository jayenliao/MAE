import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_save_path(csv_path:str, save_dir:str="auto") -> str:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if save_dir == "auto":
        save_dir = os.path.join(os.path.dirname(csv_path), "metric_plots")
        save_path = os.path.join(save_dir, "training_loss.png")
    else:
        save_path = os.path.join(save_dir, "training_loss.png")
    os.makedirs(save_dir, exist_ok=True)
    return save_path

def plot_metrics(metrics:pd.DataFrame, figsize:tuple[float, float], save_path:str):
    plt.figure(figsize=figsize)
    plt.plot(metrics['epoch'], metrics['train_loss'])
    plt.grid(linestyle='--', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing metrics.")
    parse.add_argument("--save_dir", type=str, default="auto", help="Directory to save the plot image.")
    parse.add_argument("--figsize", type=float, nargs=2, default=(4, 3), help="Figure size for the plot.")

    args = parse.parse_args()
    metrics = pd.read_csv(args.csv_path)
    save_path = get_save_path(args.csv_path, args.save_dir)
    plot_metrics(metrics, figsize=args.figsize, save_path=save_path)
