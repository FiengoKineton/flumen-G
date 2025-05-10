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
    sections_gen = {
        'VDP': {
            'old': 'src/flumen/finals/csv/VDP/old/default/old_vdp.csv',
            'lpv': 'src/flumen/finals/csv/VDP/new/vdp-lpv.csv',
            'static': 'src/flumen/finals/csv/VDP/new/vdp-static.csv',
        },
        'FHN_true': {
            'old': 'src/flumen/finals/csv/FHN/old/default/old_fhn.csv',
            'BE_stat': 'src/flumen/finals/csv/FHN/new/True/fhn-BE_stat_true.csv',
            'FE_lpv': 'src/flumen/finals/csv/FHN/new/True/fhn-FE_lpv_false_T.csv',
            'alpha_true': 'src/flumen/finals/csv/FHN/new/True/fhn-alpha-T.csv',
        },
        'FHN_false': {
            'old': 'src/flumen/finals/csv/FHN/old/default/old_fhn.csv',
            'alpha_false': 'src/flumen/finals/csv/FHN/new/fhn-new_method.csv',
            'oval_false': 'src/flumen/finals/csv/FHN/new/fhn-oval_false.csv',
        },
        'NAD': {
            'old_stable': 'src/flumen/finals/csv/NAD/old/default/old_nad-stable.csv',
            'new_stable': 'src/flumen/finals/csv/NAD/new/nad-stable.csv',
            'old_big': 'src/flumen/finals/csv/NAD/old/default/old_nad-big.csv',
            'new_big': 'src/flumen/finals/csv/NAD/new/nad-big.csv',
        }
    }

    sections_sin = {
        'VDP-sin': {
            'old_sin': 'src/flumen/finals/csv/VDP/old/sin/old_vdp-sin.csv',
            'new_sin': 'src/flumen/finals/csv/VDP/new/sin/vdp-sin.csv',
        },
        #'FHN-sin': {
            #'old_sin': 'src/flumen/finals/csv/FHN/old/sin/old_fhn-sin.csv',
            #'new_sin': 'src/flumen/finals/csv/FHN/new/sin/fhn-sin.csv',
        #},
        'NAD-sin': {
            'old_sin': 'src/flumen/finals/csv/NAD/old/sin/old_nad-stable-sin.csv',
            'new_sin': 'src/flumen/finals/csv/NAD/new/sin/nad-stable-sin.csv',
            'old_big_sin': 'src/flumen/finals/csv/NAD/old/sin/old_nad-big-sin.csv',
            'new_big_sin': 'src/flumen/finals/csv/NAD/new/sin/nad-big-sin.csv',
        }
    }

    sections_DS = {
        'vdp-DS': {
            'old': 'src/flumen/finals/csv/VDP/old/default/old_vdp.csv',
            'small_DS': 'src/flumen/finals/csv/VDP/new/small_DS/vdp-small_DS.csv',
        },
        'fhn-DS': {
            'old': 'src/flumen/finals/csv/FHN/old/default/old_fhn.csv',
            'small_DS': 'src/flumen/finals/csv/FHN/new/small_DS/fhn-small_DS.csv',
        },
        'nad-DS': {
            'old_big': 'src/flumen/finals/csv/NAD/old/default/old_nad-big.csv',
            'small_DS': 'src/flumen/finals/csv/NAD/new/small_DS/nad-big-small_DS.csv',
        }
    }

    """sections = sections_DS
    sim_metrics = SimulationMetrics(sections)
    stats = sim_metrics.compute_stats()"""

    all_sections = {
        "General": sections_gen,
        "Sinusoidal": sections_sin,
        "DS": sections_DS
    }

    for section_type, sections in all_sections.items():
        print(f"\n\n====================== {section_type} Sections ======================\n")

        sim_metrics = SimulationMetrics(sections)
        stats = sim_metrics.compute_stats()


        for section, models in stats.items():
            print("\n----------------------------------------------------------------------------")
            print("----------------------------------------------------------------------------")
            print(f"\nMetrics Comparison for {section}:")

            # Box and trend plots for section
            sim_metrics.plot_box(section)
            ###sim_metrics.plot_trend(section)

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
