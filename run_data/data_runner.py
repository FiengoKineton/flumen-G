import subprocess


general = "run_data/csv_files/wandb_get_runs.csv"       # comment | all the wandb runs  
temp = "run_data/csv_files/temp.csv"                    # comment | temporary file
models = "run_data/csv_files/models.csv"                # comment | different models

test1 = "run_data/csv_files/sweep_test1.csv"            # comment | vdp beta update
test2 = "run_data/csv_files/sweep_test2.csv"            # comment | vdp sweep_config_test_2
test3 = "run_data/csv_files/sweep_test3.csv"            # comment | fhn sweep_config_test_3
test4 = "run_data/csv_files/sweep_test4.csv"            # comment | vdp sweep_config_test_2 right dyn_model
test5 = "run_data/csv_files/sweep_test5.csv"            # comment | twotank sweep_config_test_3
test6 = "run_data/csv_files/sweep_test6.csv"            # comment | vdp sweep_config_test_4

vdp_fin = "run_data/csv_files/Finals_vdp.csv"           # comment | FINAL COMPARISON for vdp
fhn_fin = "run_data/csv_files/Finals_fhn.csv"           # comment | FINAL COMPARISON for fhn
default = "run_data/csv_files/Default_code.csv"         # comment | original RNN architecture
table = "run_data/csv_files/table.csv"                  # comment | for LateX tables
csv_path = vdp_fin


# Define your scripts and optional args
scripts_gen = [
    #{"file": "run_data/wandb_get_runs.py", "args": []},
    #{"file": "run_data/wandb_calculus_min_max.py", "args": ["--which", 'best_val']},                        # ["--loc", "--which"]
    #{"file": "experiments/hyperparams.py", "args": ["--run"]},                                              # ["--run"]
    {"file": "scr/flumen/others/dyn_plots.py", "args": ["--both", "--mhu", "-0.01", "--k", "1.0"]},                  # ["--both", "--k", "--mhu"]
]

scripts_spc = [
    {"file": "run_data/output_calculus.py", "args": ["--plot", "--display", "--loc", csv_path]},            # ["--plot", "--all", "--display", "--loc"]
    {"file": "run_data/wandb_calculus_sort.py", "args": ["--loc", csv_path]},                               # ["--loc", "--all"]
    #{"file": "run_data/wandb_calculus_plots.py", "args": ["--loc", csv_path]},                              # ["--all", "--loc"]
]


scripts = scripts_spc

print("\ncsv_path:", csv_path, "\nscripts:", scripts, "\n\n")

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
