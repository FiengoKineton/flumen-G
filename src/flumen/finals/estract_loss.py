import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re


# ---------------------------------------------------------
# Init

CSV = False
PLOT = False
ALL = False

which = "fhn"


sets = {
    'fhn': {
        'model_name': 'FitzHugh-Nagumo', 
        'n': 2, 
        'Note' : {
            'name_1': 'fhn-old',        # FHN-old
            'name_2': 'fhn',            # fhn-upd_steps
        }
    }, 
    'vdp': {
        'model_name': 'Van der Pol', 
        'n': 2, 
        'Note' : {
            'name_1': 'vdp-old',        # VDP-old
            'name_2': 'vdp-lpv',        # VDP-lpv
            'name_3': 'vdp-static',     # VDP-static
        }
    }, 
    'nad_big': {
        'model_name': 'NonlinearActivationDynamics', 
        'n': 15, 
        'Note' : {
            'name_1': 'nad_big-old',    # NAD_big-old   
            'name_2': 'nad_big',        # NAD_big
        }
    }, 
    'nad_small': {
        'model_name': 'NonlinearActivationDynamics', 
        'n': 5, 
        'Note' : {
            'name_1': 'nad_small-old',  # NAD_small-old
            'name_2': 'nad_small',      # NAD_small
        }
    }
}


estract = sets[which]
name = estract["Note"]["name_2"]
name_dir = f'src/flumen/finals/csv/loss/{name}.csv'

# ---------------------------------------------------------
# Functions

def create_csv(): 
    # Load the log file
    with open("src/flumen/finals/output.log", "r") as f:
        log_data = f.readlines()

    # Filter lines that match the pattern: epoch number followed by values
    pattern = re.compile(r'^\s*(\d+)\s+::\s+([\deE\+\-\.]+)\s+::\s+([\deE\+\-\.]+)\s+::\s+([\deE\+\-\.]+)\s+::\s+([\deE\+\-\.]+)')
    rows = [pattern.match(line).groups() for line in log_data if pattern.match(line)]

    # Create DataFrame
    df = pd.DataFrame(rows, columns=["n", "train", "val", "test", "best_val"])

    # Convert numeric columns
    df = df.astype({
        "n": int,
        "train": float,
        "val": float,
        "test": float,
        "best_val": float
    })

    df.to_csv(name_dir, index=False)
    # import ace_tools as tools; tools.display_dataframe_to_user(name="Filtered Training Log CSV", dataframe=df)

def plot_loss(): 
    df = pd.read_csv(name_dir)

    # === PLOT ===
    plt.figure(figsize=(10, 6))
    line1, = plt.plot(df["n"], df["train"], label="Train Loss", linewidth=2)
    line2, = plt.plot(df["n"], df["val"], label="Validation Loss", linewidth=2)
    line3, = plt.plot(df["n"], df["test"], label="Test Loss", linewidth=2)

    # Best val
    best_val_final = df["best_val"].iloc[-1]
    first_best_epoch = df[df["val"] == best_val_final]["n"].iloc[0]

    # Linea verticale tratteggiata
    label_best = f'Best Val @ Epoch {first_best_epoch} ({best_val_final:.4e})'
    line4 = plt.axvline(x=first_best_epoch, color='gray', linestyle='--', linewidth=1,
                        label=label_best)


    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f'{estract["model_name"]} dynamics ($x \\in \\mathcal{{R}}^{{{estract["n"]}}}$) -- {name}')
    plt.legend(handles=[line1, line2, line3, line4], loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.yscale('log')

    # Show or save
    if which=="vdp":    loc = 'VDP'
    elif which=="fhn":  loc = 'FHN'
    else:               loc = 'NAD'
    plt.savefig(f'src/flumen/finals/pdf/{loc}/others/loss_{name}.pdf', dpi=300)
    plt.show()

def plot_model_set():
    notes = estract["Note"]
    model_name = estract["model_name"]
    n_dim = estract["n"]

    plt.figure(figsize=(10, 6))

    # Base colors
    base_colors = {
        'train': 'tab:blue',
        'val': 'tab:orange',
        'test': 'tab:green'
    }

    # Tonalità: light, medium, dark
    tone_opacity = {
        0: 0.50, # name_1 (old) - light
        1: 0.75, # name_2       - medium
        2: 1.00  # name_3       - full
    }

    for idx, key in enumerate(notes):
        name_i = notes[key]
        path = f'src/flumen/finals/csv/loss/{name_i}.csv'

        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"⚠️  CSV not found: {path}")
            continue

        # Plot each loss type with adjusted alpha
        for loss_type in ['train', 'val', 'test']:
            linestyle = {'train': '-', 'val': '--', 'test': ':'}[loss_type]
            plt.plot(
                df["n"],
                df[loss_type],
                linestyle=linestyle,
                color=base_colors[loss_type],
                alpha=tone_opacity[idx],
                label=f'{loss_type.capitalize()} - {name_i}'
            )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f'{model_name} dynamics ($x \\in \\mathcal{{R}}^{{{n_dim}}}$)')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.yscale('log')

    if which == "vdp":    loc = 'VDP'
    elif which == "fhn":  loc = 'FHN'
    else:                 loc = 'NAD'

    plt.savefig(f'src/flumen/finals/pdf/{loc}/others/loss_comparison_{which}.pdf', dpi=300)
    plt.show()

# ---------------------------------------------------------
# Runs

if CSV:     create_csv()
if PLOT:    plot_loss()
if ALL:     plot_model_set()
