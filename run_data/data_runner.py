import subprocess


general = "run_data/csv_files/wandb_get_runs.csv"  
temp = "run_data/csv_files/temp.csv"  
models = "run_data/csv_files/models.csv"

test1 = "run_data/csv_files/sweep_test1.csv"            # comment | vdp beta update
test2 = "run_data/csv_files/sweep_test2.csv"            # comment | vdp sweep_config_test_2
test3 = "run_data/csv_files/sweep_test3.csv"            # comment | fhn sweep_config_test_3
test4 = "run_data/csv_files/sweep_test4.csv"            # comment | vdp sweep_config_test_2 right dyn_model
test5 = "run_data/csv_files/sweep_test5.csv"            # comment | twotank sweep_config_test_3

csv_path = test3
print(f"csv_path: {csv_path}\n")

# Define your scripts and optional args
scripts_gen = [
    {"file": "run_data/wandb_get_runs.py", "args": []},
    {"file": "run_data/wandb_calculus_min_max.py", "args": ["--which", 'val_loss']},                        # ["--loc", "--which"]
    {"file": "experiments/hyperparams.py", "args": ["--run"]},                                              # ["--run"]
]

scripts_spc = [
    {"file": "run_data/output_calculus.py", "args": ["--plot", "--display", "--loc", csv_path]},            # ["--plot", "--all", "--display", "--loc"]
    {"file": "run_data/wandb_calculus_sort.py", "args": ["--loc", csv_path, "--all"]},                      # ["--loc", "--all"]
    {"file": "run_data/wandb_calculus_plots.py", "args": ["--loc", csv_path]},                              # ["--all", "--loc"]
]


scripts = scripts_spc


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
