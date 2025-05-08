import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

class SimulationMetrics:
    def __init__(self, sections):
        """self.sections = {
            key: {name: self._parse_csv(path) for name, path in paths.items()}
            for key, paths in sections.items()
        }"""
        self.sections = {}
        for section_name, paths in sections.items():
            self.sections[section_name] = {}
            for model_name, path in paths.items():
                df = self._parse_csv(path)
                df['model_name'] = model_name  # <--- QUI il fix
                self.sections[section_name][model_name] = df
        self.metrics = ["_step", "best_val", "best_epoch", "best_test", "best_train", "time"]


    def _parse_csv(self, path):
        df = pd.read_csv(path)
        df['summary'] = df['summary'].apply(ast.literal_eval)
        summary_df = df['summary'].apply(pd.Series)
        df = pd.concat([df, summary_df], axis=1)
        return df

    def compute_stats(self):
        stats = {}
        for section, dfs in self.sections.items():
            stats[section] = {}
            for name, df in dfs.items():
                stats[section][name] = {
                    metric: {
                        'mean': df[metric].mean(),
                        'std': df[metric].std()
                    }
                    for metric in self.metrics if metric in df.columns
                }
        return stats

    def get_best_simulation(self, section):
        best_simulation = {}
        for name, df in self.sections[section].items():
            best_row = df.loc[df['best_val'].idxmin()]
            best_simulation[name] = best_row#['name']
        return best_simulation

    def plot_comparison(self, section, model_names):
        dfs = [self.sections[section][name] for name in model_names]
        df_filtered = pd.concat(dfs).copy()
        metrics = ["_step", "best_val"]

        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 10))

        for metric in metrics:
            if metric in df_filtered.columns:
                plt.subplot(2, 1, metrics.index(metric) + 1)
                sns.barplot(x='name', y=metric, data=df_filtered, errorbar='sd') #ci='sd')
                plt.title(f'{metric} Comparison - {section}')
                plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_box(self, section):
        # Aggiungi una colonna 'model_name' a ciascun dataframe
        dfs = []
        for model_name, df in self.sections[section].items():
            df_copy = df.copy()
            df_copy["model_name"] = model_name
            dfs.append(df_copy)

        df_all = pd.concat(dfs, ignore_index=True)
        metrics = ["_step", "best_val"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))

        for i, metric in enumerate(metrics):
            sns.boxplot(x='model_name', y=metric, data=df_all, ax=axes[i])
            axes[i].set_title(f'{section} - {metric} (Boxplot)')
            axes[i].set_yscale('log')

        plt.tight_layout()
        plt.show()

    def plot_trend(self, section):
        df_all = pd.concat(self.sections[section].values(), ignore_index=True)
        metrics = ["_step", "best_val"]
        plt.figure(figsize=(12, 6))
        
        for metric in metrics:
            for model in df_all['model_name'].unique():
                df_model = df_all[df_all['model_name'] == model]
                plt.plot(df_model.index, df_model[metric], marker='o', label=f'{model} - {metric}')
        
        plt.title(f'Trend of Metrics Over Runs - {section}')
        plt.xlabel("")
        plt.xticks([])
        plt.ylabel('Metric Value')
        plt.yscale('log')
        plt.legend(title='Model - Metric')
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    sections = {
        'VDP': {
            'old': 'src/flumen/finals/csv/old_vdp.csv',
            'lpv': 'src/flumen/finals/csv/vdp-lpv.csv',
            'static': 'src/flumen/finals/csv/vdp-static.csv',
        },
        'FHN': {
            'old': 'src/flumen/finals/csv/old_fhn.csv',
            'BE_stat': 'src/flumen/finals/csv/fhn-BE_stat_true.csv',
            'FE_lpv': 'src/flumen/finals/csv/fhn-FE_lpv_false_T.csv',
            'oval_false': 'src/flumen/finals/csv/fhn-oval_false.csv',
        },
        'NAD': {
            'old_stable': 'src/flumen/finals/csv/old_nad-stable.csv',
            'new_stable': 'src/flumen/finals/csv/nad-stable.csv',
            'old_big': 'src/flumen/finals/csv/old_nad-big.csv',
            'new_big': 'src/flumen/finals/csv/nad-big.csv',
        },
        'NAD-sin': {
            'old_sin': 'src/flumen/finals/csv/old_nad-stable-sin.csv',
            'new_sin': 'src/flumen/finals/csv/nad-stable-sin.csv',
            'old_big_sin': 'src/flumen/finals/csv/old_nad-big-sin.csv',
            'new_big_sin': 'src/flumen/finals/csv/nad-big-sin.csv',
        }
    }

    sim_metrics = SimulationMetrics(sections)
    stats = sim_metrics.compute_stats()


    for section, models in stats.items():
        print("\n----------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------")
        print(f"\nMetrics Comparison for {section}:")

        # Box and trend plots for section
        sim_metrics.plot_box(section)
        sim_metrics.plot_trend(section)

        # Create table-like structure
        metrics_df = pd.DataFrame(index=["Best Simulation"] + sim_metrics.metrics)

        best_simulations = sim_metrics.get_best_simulation(section)
        for model_name, best_row in best_simulations.items():
            best_sim, best_val = best_row['name'], best_row['best_val']
            metrics_df.loc["Best Simulation", model_name] = f"{best_sim} (val={best_val:.4g})"

        for model_name, metrics in models.items():
            for metric, values in metrics.items():
                metrics_df.loc[metric, model_name] = f"{values['mean']:.4f} ± {values['std']:.4f}"

        print(metrics_df.to_string())
    print("\n----------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------")

    """ # Plot comparisons for all sections
    sim_metrics.plot_comparison('VDP', ['old', 'lpv', 'static'])
    sim_metrics.plot_comparison('FHN', ['old', 'new'])
    sim_metrics.plot_comparison('NAD', ['old_stable', 'new_stable', 'old_big', 'new_big'])
    # """




    """
    VDP best runs: 
    - old | vdp_fin-old-2: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-x3zk3ip4:v0
    - lpv | vdp_fin-3: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-lbz1tnpu:v3
    - static | vdp_fin-25: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-lwqp2l3z:v3

    (vdp_lpv.pdf) python experiments/interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-lbz1tnpu:v3 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-x3zk3ip4:v0
    (vdp_static.pdf) python experiments/interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-lwqp2l3z:v3 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-x3zk3ip4:v0
    (vdp_sin.pdf)  python.exe .\experiments\interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-22h9jfjb:v0 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-sjelftdk:v0

    Note: (decoder_mode, linearisation_mode, discretisation_mode) = [(False, lpv, TU) and (False, static, TU)]
    -------------------------------------------------------

    FHN best runs: (fhn--32)
    - old | fhn_fin-old-2: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-w52hqxkd:v1
    - BE_stat | fhn--04: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-04y8vw0k:v4
    - FE_lpv | fhn--12: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-vu6o3roj:v2
    - oval | fhn_swift-r=2--3: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-rystn8ww:v4

    (fhn_stat.pdf) python experiments/interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-04y8vw0k:v4 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-w52hqxkd:v1
    (fhn_lpv.pdf) python experiments/interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-vu6o3roj:v2 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-w52hqxkd:v1
    (fhn_oval.pdf) python experiments/interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-rystn8ww:v4 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-w52hqxkd:v1

    Note: (decoder_mode, linearisation_mode, discretisation_mode) = (True, static, BE)
    PS: change to True self.decoder_mode in model.py for FE_lpv and set swift=0 and use Circle
    PPS: use Elipse and set swift=1 for oval
    -------------------------------------------------------
    
    NAD best runs: 
    - old_stable | nad_fin-old-2: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-mg4z6swx:v1
    - new_stable | nad_fin-01: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-zshs5333:v0
    - old_big | nad_fin-old-big: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-jwlwuqmw:v0
    - new_big | nad_big_fin-05: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-3dxiz9gf:v2
    
    (nad_stable.pdf) python experiments/interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-zshs5333:v0 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-mg4z6swx:v1
    (nad_big.pdf) python experiments/interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-3dxiz9gf:v2 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-jwlwuqmw:v0
    
    Note: (decoder_mode, linearisation_mode, discretisation_mode) = (False, static, FE)
    PS: change nad.yaml in section [state_dim, mode]
    -------------------------------------------------------
    
    NAD-sin best runs: 
    - old_stable | nad_sin_old-02: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-ux5pkc68:v2
    - new_stable | nad_sin-02(03): (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-xvc3v8nf:v0
    - old_big | nad_sin_big_old-01: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-rm7wha0k:v0
    - new_big | nad_sin_big-02: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-orysyc0y:v1
    
    (nad_sin.pdf) python experiments/interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-xvc3v8nf:v0 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-ux5pkc68:v2
    (nad_big_sin.pdf) python experiments/interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-orysyc0y:v1 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-rm7wha0k:v0
    
    Note: (decoder_mode, linearisation_mode, discretisation_mode) = (False, static, FE)
    PS: change nad.yaml in section [state_dim, mode]
    -------------------------------------------------------
    -------------------------------------------------------
    
    save it in pdf in ./src/flumen/finals/

    run: python experiments/interactive_test_compare.py --wandb (new) --wandb_2 (old)
    
    -------------------------------------------------------
    
    go here to see wandb report: https://wandb.ai/aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/reports/Finals--VmlldzoxMjM5MjEwNA
    
    

    RESULTs
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    Metrics Comparison for VDP:
                                            old                      lpv                    static
    Best Simulation  vdp_fin-old-2 (val=0.09267)  vdp_fin-3 (val=0.03983)  vdp_fin-25 (val=0.03364)
    _step                      71.0000 ± 39.5980        52.1818 ± 21.1699         52.8889 ± 24.0907
    best_val                     0.0937 ± 0.0015          0.0576 ± 0.0161           0.0524 ± 0.0140
    best_epoch                 46.0000 ± 39.5980        31.7273 ± 20.2044         32.0000 ± 25.7294
    best_test                    0.2186 ± 0.2193          0.1256 ± 0.0730           0.1546 ± 0.1022
    best_train                   0.0168 ± 0.0148          0.0223 ± 0.0163           0.0264 ± 0.0249
    time                 28914.4827 ± 26327.1509  43529.4471 ± 17019.0007    25954.0377 ± 9905.1619

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    Metrics Comparison for FHN:
                                            old                  BE_stat                   FE_lpv                      oval_false
    Best Simulation  fhn_fin-old-2 (val=0.01486)    fhn--04 (val=0.01476)    fhn--12 (val=0.01339)  fhn_swift-r=2--3 (val=0.04399)
    _step                      131.0000 ± 7.0711        88.1111 ± 21.5954        43.7500 ± 19.9228               76.8889 ± 25.0272
    best_val                     0.0167 ± 0.0027          0.0239 ± 0.0057          0.0314 ± 0.0147                 0.0649 ± 0.0195
    best_epoch                 106.0000 ± 7.0711        66.6667 ± 26.9676        42.2500 ± 19.3628               61.8889 ± 25.0272
    best_test                    0.0258 ± 0.0141          0.0337 ± 0.0086          0.0376 ± 0.0074                 0.0773 ± 0.0190
    best_train                   0.0026 ± 0.0004          0.0069 ± 0.0063          0.0175 ± 0.0102                 0.0416 ± 0.0179
    time                 40958.6990 ± 20805.0690  60356.5976 ± 18251.3046  45478.9708 ± 26353.0884        129533.3689 ± 66097.3804

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    Metrics Comparison for NAD:
                                      old_stable                  new_stable                        old_big                       new_big
    Best Simulation  nad_fin-old-2 (val=0.001105)  nad_fin-01 (val=0.0006947)  nad_fin-old-big (val=0.02377)  nad_big_fin-05 (val=0.01123)
    _step                      151.6667 ± 83.7158            46.7500 ± 8.8761             144.0000 ± 96.9948             53.1250 ± 23.3510
    best_val                      0.0023 ± 0.0014             0.0012 ± 0.0003                0.0340 ± 0.0103               0.0138 ± 0.0024
    best_epoch                 139.0000 ± 94.3981           26.1250 ± 12.0646            135.3333 ± 111.1411             42.5000 ± 22.8661
    best_test                     0.0033 ± 0.0031             0.0012 ± 0.0004                0.0264 ± 0.0080               0.0131 ± 0.0009
    best_train                    0.0003 ± 0.0001             0.0004 ± 0.0001                0.0020 ± 0.0028               0.0007 ± 0.0002
    time                  50041.6461 ± 17418.2742      32057.6465 ± 6607.2100        37678.3957 ± 30192.6393       41588.0502 ± 23762.2955

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    Metrics Comparison for NAD-sin:
                                        old_sin                    new_sin                       old_big_sin                   new_big_sin      
    Best Simulation  nad_sin_old-02 (val=0.002201)  nad_sin-02 (val=0.002334)  nad_sin_big_old-01 (val=0.03625)  nad_sin_big-02 (val=0.01416)      
    _step                       137.5000 ± 65.7609           41.8000 ± 6.7231                  30.5000 ± 0.7071             68.8000 ± 25.0140      
    best_val                       0.0034 ± 0.0017            0.0035 ± 0.0011                   0.0398 ± 0.0051               0.0170 ± 0.0020      
    best_epoch                  112.5000 ± 65.7609           20.8000 ± 7.5631                   5.5000 ± 0.7071             47.8000 ± 19.9424      
    best_test                      0.0074 ± 0.0026            0.0032 ± 0.0013                   0.0422 ± 0.0042               0.0173 ± 0.0012      
    best_train                     0.0007 ± 0.0001            0.0009 ± 0.0001                   0.0069 ± 0.0012               0.0016 ± 0.0003      
    time                   97554.6680 ± 52218.0499    42605.6057 ± 22395.0561             24689.9393 ± 270.9701       71497.5481 ± 49352.4864      

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    """