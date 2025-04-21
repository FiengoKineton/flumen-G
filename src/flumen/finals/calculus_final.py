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
            best_simulation[name] = best_row['name']
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
            'new': 'src/flumen/finals/csv/fhn.csv',
        },
        'NAD': {
            'old_stable': 'src/flumen/finals/csv/old_nad-stable.csv',
            'new_stable': 'src/flumen/finals/csv/nad-stable.csv',
            'old_big': 'src/flumen/finals/csv/old_nad-big.csv',
            'new_big': 'src/flumen/finals/csv/nad-big.csv',
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
        for model_name, best_sim in best_simulations.items():
            metrics_df.loc["Best Simulation", model_name] = best_sim

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
    
    Note: (decoder_mode, linearisation_mode, discretisation_mode) = [(False, lpv, TU) and (False, static, TU)]
    -------------------------------------------------------

    FHN best runs: 
    - old | fhn_fin-old-2: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-w52hqxkd:v1
    - new | fhn--04: (wandb) aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-04y8vw0k:v4

    (fhn.pdf) python experiments/interactive_test_compare.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-04y8vw0k:v4 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-w52hqxkd:v1
    
    Note: (decoder_mode, linearisation_mode, discretisation_mode) = (True, static, BE)
    PS: change to True self.decoder_mode in model.py
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
    -------------------------------------------------------
    
    save it in pdf in ./src/flumen/finals/

    run: python experiments/interactive_test_compare.py --wandb (new) --wandb_2 (old)
    
    -------------------------------------------------------
    
    go here to see wandb report: https://wandb.ai/aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/reports/Finals--VmlldzoxMjM5MjEwNA
    """