import subprocess


general = "run_data/csv_files/wandb_get_runs.csv"       # comment | all the wandb runs  
___temp = "run_data/csv_files/temp.csv"                 # comment | temporary file
_models = "run_data/csv_files/models.csv"               # comment | different models

__test1 = "run_data/csv_files/sweep_test1.csv"          # comment | vdp beta update
__test2 = "run_data/csv_files/sweep_test2.csv"          # comment | vdp sweep_config_test_2
__test3 = "run_data/csv_files/sweep_test3.csv"          # comment | fhn sweep_config_test_3
__test4 = "run_data/csv_files/sweep_test4.csv"          # comment | vdp sweep_config_test_2 right dyn_model
__test5 = "run_data/csv_files/sweep_test5.csv"          # comment | twotank sweep_config_test_3
__test6 = "run_data/csv_files/sweep_test6.csv"          # comment | vdp sweep_config_test_4

vdp_fin = "run_data/csv_files/Finals_vdp.csv"           # comment | FINAL COMPARISON for vdp
fhn_fin = "run_data/csv_files/Finals_fhn.csv"           # comment | FINAL COMPARISON for fhn
nad_fin = "run_data/csv_files/Finals_nad.csv"           # comment | FINAL COMPARISON for nad
default = "run_data/csv_files/Default_code.csv"         # comment | original RNN architecture
__table = "run_data/csv_files/table.csv"                # comment | for LateX tables


csv_path = fhn_fin                                      # "run_data/txt_files/temp.csv"

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



"""
csv_path: run_data/csv_files/models.csv 
scripts: [{'file': 'run_data/wandb_calculus_sort.py', 'args': ['--loc', 'run_data/csv_files/models.csv']}] 



▶️      Running Script 1: run_data/wandb_calculus_sort.py
✅      Script 1 Output:

---------------------------------------
Order by min:
---------------------------------------


Top 20 results by best_val:
           val_loss  best_val  best_test  best_train  epoch   
nad-s-RK4  0.000502  0.000496   0.000456    0.000406     78 19
 nad-s-TU  0.000531  0.000504   0.000458    0.000407     86 18
 nad-s-FE  0.000540  0.000529   0.000474    0.000411     80 14
 nad-m-FE  0.000701  0.000700   0.000837    0.000437    102 16
 nad-m-TU  0.000811  0.000811   0.000969    0.000551     39 20
nad-m-old  0.000908  0.000906   0.001386    0.000469    136 17
nad-s-old  0.000943  0.000937   0.000705    0.000463    155 15


Top 20 results by epoch:
           val_loss  best_val  best_test  best_train  epoch
 nad-m-TU  0.000811  0.000811   0.000969    0.000551     39 20
nad-s-RK4  0.000502  0.000496   0.000456    0.000406     78 19
 nad-s-FE  0.000540  0.000529   0.000474    0.000411     80 14
 nad-s-TU  0.000531  0.000504   0.000458    0.000407     86 18
 nad-m-FE  0.000701  0.000700   0.000837    0.000437    102 16
nad-m-old  0.000908  0.000906   0.001386    0.000469    136 17
nad-s-old  0.000943  0.000937   0.000705    0.000463    155 15


Top 20 results by weighted score:
           best_val  best_test  best_train  epoch     score
nad-s-RK4  0.000496   0.000456    0.000406     78  5.148892 19
 nad-s-TU  0.000504   0.000458    0.000407     86  5.311501 18
 nad-s-FE  0.000529   0.000474    0.000411     80  5.380265 14
 nad-m-FE  0.000700   0.000837    0.000437    102  7.425654 16
 nad-m-TU  0.000811   0.000969    0.000551     39  7.481617 20
nad-s-old  0.000937   0.000705    0.000463    155  8.999464 15
nad-m-old  0.000906   0.001386    0.000469    136 10.287037 17
"""