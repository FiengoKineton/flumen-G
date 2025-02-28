import os
import pandas as pd
import matplotlib.pyplot as plt


def get_latest_files():
    """Find the latest dataset for each optimizer in DT_comparison."""
    
    # Define the folder relative to the `experiments` directory
    folder = os.path.join(os.path.dirname(__file__), "DT_comparison")
    
    if not os.path.exists(folder):
        print(f"No comparison data found in {folder}")
        return {}

    files = os.listdir(folder)
    optimizer_files = {}

    for file in files:
        if file.endswith(".csv"):
            optimizer, num = file.rsplit("_", 1)
            num = num.split(".")[0]  # Remove .csv extension

            if num.isdigit():
                num = int(num)
                if optimizer not in optimizer_files or num > optimizer_files[optimizer][1]:
                    optimizer_files[optimizer] = (os.path.join(folder, file), num)

    return {opt: path for opt, (path, _) in optimizer_files.items()}


def plot_comparison(dataframes):
    """Plot training loss and validation loss for different optimizers."""
    plt.figure(figsize=(10, 5))

    # Training Loss
    for opt, df in dataframes.items():
        plt.plot(df["epoch"], df["train_loss"], label=f"{opt} - x1")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.title("Train Loss Comparison")
    plt.show()

    # Validation Loss
    plt.figure(figsize=(10, 5))
    for opt, df in dataframes.items():
        plt.plot(df["epoch"], df["val_loss"], label=f"{opt} - x2")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.title("Validation Loss Comparison")
    plt.show()

    # Training Time per Epoch
    plt.figure(figsize=(10, 5))
    for opt, df in dataframes.items():
        plt.plot(df["epoch"], df["epoch_time"], label=f"{opt} - Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.title("Time per Epoch Comparison")
    plt.show()

def main():
    latest_files = get_latest_files()

    if not latest_files:
        print("No datasets available for comparison.")
        return

    # Load data
    dataframes = {opt: pd.read_csv(path) for opt, path in latest_files.items()}

    print("\nComparing the following optimizer runs:")
    for opt, path in latest_files.items():
        print(f"- {opt}: {path}")

    plot_comparison(dataframes)

if __name__ == "__main__":
    main()  # Ensure the script only runs if executed from the terminal
