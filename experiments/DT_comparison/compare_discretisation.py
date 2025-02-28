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


def plot_discretization_comparison(dataframes):
    """Plot x1, x2 values and computation time for different discretization methods."""
    
    # Plot x1 evolution over epochs
    plt.figure(figsize=(10, 5))
    for disc, df in dataframes.items():
        plt.plot(df["epoch"], df["x1"], label=f"{disc} - x1")
    plt.xlabel("Epoch")
    plt.ylabel("x1 Value")
    plt.legend()
    plt.title("x1 Evolution Comparison")
    plt.show()
 
    # Plot x2 evolution over epochs
    plt.figure(figsize=(10, 5))
    for disc, df in dataframes.items():
        plt.plot(df["epoch"], df["x2"], label=f"{disc} - x2")
    plt.xlabel("Epoch")
    plt.ylabel("x2 Value")
    plt.legend()
    plt.title("x2 Evolution Comparison")
    plt.show()
 
    # Plot computation time per epoch
    plt.figure(figsize=(10, 5))
    for disc, df in dataframes.items():
        plt.plot(df["epoch"], df["epoch_time"], label=f"{disc} - Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.title("Computation Time per Epoch (Discretization Methods)")
    plt.show()
 
 
def main():
    # Load the latest results for discretization comparisons
    latest_discretization_files = get_latest_discretization_files()
 
    if not latest_discretization_files:
        print("No datasets available for discretization comparison.")
        return
 
    # Load data
    discretization_data = {disc: pd.read_csv(path) for disc, path in latest_discretization_files.items()}
 
    # Print information
    print("\nComparing the following discretization runs:")
    for disc, path in latest_discretization_files.items():
        print(f"- {disc}: {path}")
 
    # Plot discretization comparisons
    plot_discretization_comparison(discretization_data)
 
 
if __name__ == "__main__":
    main()
