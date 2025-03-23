import subprocess


file_path_1 = "run_data/csv_files/wandb_get_runs.csv"  # Use the correct relative path
file_path_2 = "run_data/csv_files/temp.csv"  # Use the correct relative path
file_path_3 = "run_data/csv_files/sweep_test1.csv"
file_path_4 = "run_data/csv_files/sweep_test2.csv"
file_path_5 = "run_data/csv_files/models.csv"
file_path_6 = "run_data/csv_files/sweep_test3.csv"
file_path_7 = "run_data/csv_files/sweep_test4.csv"

csv_path = file_path_7


# Define your scripts and optional args
scripts = [
    #{"file": "run_data/wandb_get_runs.py", "args": []},
    {"file": "run_data/output_calculus.py", "args": ["--plot", "--display", "--loc", csv_path]},            # ["--plot", "--all", "--display", "--loc"]
    {"file": "run_data/wandb_calculus_sort.py", "args": ["--loc", csv_path, "--all"]},                      # ["--loc", "--all"]
    {"file": "run_data/wandb_calculus_plots.py", "args": ["--all", "--loc", csv_path]},                     # ["--all", "--loc"]
    {"file": "run_data\wandb_calculus_min_max.py", "args": ["--which", 'train_loss']},                        # ["--loc", "--which"]
    #{"file": "experiments/hyperparams.py", "args": ["--run"]},                                              # ["--run"]
]


# Run each script
for i, script in enumerate(scripts, 1):
    print(f"\n▶️\tRunning Script {i}: {script['file']}")
    try:
        result = subprocess.run(
            ["python", script["file"]] + script["args"],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅\tScript {i} Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌\tScript {i} failed with error:\n{e.stderr}")
