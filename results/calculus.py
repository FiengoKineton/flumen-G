import pandas as pd
import matplotlib.pyplot as plt



class CalcValues:
    def __init__(self):
        self.datasets = self.DataSet()

        self.metrics = [
            "_step", 
            "best_epoch", 
            "best_test", 
            "best_train", 
            "best_val", 
            "lr", 
            "test_loss", 
            "time", 
            "train_loss", 
            "val_loss"
            ]

        for name, data in self.datasets.items():    self.display_results(name, data)
        self.display_final_comparison()



    def DataSet(self):
        default_code_same_dim = [
            {'run': "017", '_runtime': 7546.45702, '_step': 117, '_timestamp': 1741182593.7156532, '_wandb': {'runtime': 7546}, 'best_epoch': 98, 'best_test': 0.03409814938075013, 'best_train': 0.018149984931504284, 'best_val': 0.028380416333675385, 'epoch': 118, 'lr': 0.000125, 'test_loss': 0.03472252781428988, 'time': 7528.629385232925, 'train_loss': 0.015891212812334143, 'val_loss': 0.03212327557423758},
            {'run': "026", '_runtime': 7346.7665658, '_step': 106, '_timestamp': 1741430459.320042, '_wandb': {'runtime': 7346}, 'best_epoch': 87, 'best_test': 0.025843684265892657, 'best_train': 0.014283652101993245, 'best_val': 0.02398856176388642, 'epoch': 107, 'lr': 0.000125, 'test_loss': 0.026321270325708957, 'time': 7328.914924144745, 'train_loss': 0.014671892942811446, 'val_loss': 0.026048427477242456},
            {'run': "027", '_runtime': 8290.5964382, '_step': 126, '_timestamp': 1741431693.8628545, '_wandb': {'runtime': 8290}, 'best_epoch': 107, 'best_test': 0.027500419862686643, 'best_train': 0.01666778990772193, 'best_val': 0.05470035585855681, 'epoch': 127, 'lr': 0.00025, 'test_loss': 0.027228796898963903, 'time': 8272.850373268127, 'train_loss': 0.014251226118258223, 'val_loss': 0.06482056906772038}
        ]

        x_update_alpha = [
            {'run': "015", '_runtime': 11637.674524, '_step': 65, '_timestamp': 1741122288.162348, '_wandb': {'runtime': 11637}, 'best_epoch': 56, 'best_test': 0.15566413057228876, 'best_train': 0.03127661450869507, 'best_val': 0.0726061523670242, 'epoch': 66, 'lr': 0.001, 'test_loss': 0.17950638903984947, 'time': 11612.085956335068, 'train_loss': 0.0874257475413658, 'val_loss': 0.16335126840405997},
            {'run': "018(1)", '_runtime': 12845.8606273, '_step': 84, '_timestamp': 1741201116.3583372, '_wandb': {'runtime': 12845}, 'best_epoch': 80, 'best_test': 0.09021028982741491, 'best_train': 0.020383733487318433, 'best_val': 0.055468578689864705, 'epoch': 85, 'lr': 0.001, 'test_loss': 0.11940238163584754, 'time': 12816.142216682434, 'train_loss': 0.02383146670563196, 'val_loss': 0.0593999158591032},
            {'run': "018(2-3)", '_runtime': 29190.6178727, '_step': 156, '_timestamp': 1741217461.1155827, '_wandb': {'runtime': 29190}, 'best_epoch': 137, 'best_test': 0.08476408736573325, 'best_train': 0.01709923068327563, 'best_val': 0.04871034687237134, 'epoch': 157, 'lr': 0.000125, 'test_loss': 0.08193321597008478, 'time': 29160.899462223053, 'train_loss': 0.01672123080838925, 'val_loss': 0.04979510998560323},
            {'run': "019", '_runtime': 23343.4636111, '_step': 142, '_timestamp': 1741280362.9845507, '_wandb': {'runtime': 23343}, 'best_epoch': 123, 'best_test': 0.0533087533558645, 'best_train': 0.02122072688249684, 'best_val': 0.25057389338811237, 'epoch': 143, 'lr': 6.25e-05, 'test_loss': 0.05645283382563364, 'time': 23315.39107489586, 'train_loss': 0.01923207267034779, 'val_loss': 0.25213538606961566},
            {'run': "020", '_runtime': 19282.2622103, '_step': 109, '_timestamp': 1741293582.378608, '_wandb': {'runtime': 19282}, 'best_epoch': 90, 'best_test': 0.1449957171839381, 'best_train': 0.02281889265708665, 'best_val': 0.03265471753501703, 'epoch': 110, 'lr': 0.000125, 'test_loss': 0.14242024950328327, 'time': 19247.294664382935, 'train_loss': 0.019359053211119123, 'val_loss': 0.03382213047099492},
            {'run': "021", '_runtime': 24220.1446849, '_step': 145, '_timestamp': 1741321179.491952, '_wandb': {'runtime': 24220}, 'best_epoch': 126, 'best_test': 0.06649408028239295, 'best_train': 0.018753742099439027, 'best_val': 0.11177096456762343, 'epoch': 146, 'lr': 3.125e-05, 'test_loss': 0.06538293534328067, 'time': 24190.640423059464, 'train_loss': 0.017817277136066602, 'val_loss': 0.1155335587405023}
        ]

        x_update_alpha_opt = [
            {'run': "022", '_runtime': 15588.0104955, '_step': 79, '_timestamp': 1741373321.7143245, '_wandb': {'runtime': 15588}, 'best_epoch': 60, 'best_test': 0.19954035431146624, 'best_train': 0.033677110980663984, 'best_val': 0.1268843865347287, 'epoch': 80, 'lr': 0.0005, 'test_loss': 0.17315763891452834, 'time': 15557.825784683228, 'train_loss': 0.02871229724771289, 'val_loss': 0.13715681126193394}, 
            {'run': "023", '_runtime': 19091.3150562, '_step': 90, '_timestamp': 1741383918.291978, '_wandb': {'runtime': 19091}, 'best_epoch': 71, 'best_test': 0.17176606902290906, 'best_train': 0.03029533599813779, 'best_val': 0.0722720061857549, 'epoch': 91, 'lr': 0.00025, 'test_loss': 0.15187193074869731, 'time': 19058.01224398613, 'train_loss': 0.021802850865892003, 'val_loss': 0.08129255662834833},
            {'run': "024", '_runtime': 16780.6603587, '_step': 74, '_timestamp': 1741391706.1888623, '_wandb': {'runtime': 16780}, 'best_epoch': 55, 'best_test': 0.29205070069384953, 'best_train': 0.030286846978087274, 'best_val': 0.058476886756363367, 'epoch': 75, 'lr': 0.00025, 'test_loss': 0.20884731518370764, 'time': 16748.41609811783, 'train_loss': 0.024770572751997007, 'val_loss': 0.06078948893599094},
            {'run': "025", '_runtime': 6466.9516759, '_step': 47, '_timestamp': 1741404462.7291248, '_wandb': {'runtime': 6466}, 'best_epoch': 28, 'best_test': 0.2853936068122349, 'best_train': 0.11861003164655318, 'best_val': 0.1437250425418218, 'epoch': 48, 'lr': 0.0005, 'test_loss': 0.3043967697118956, 'time': 6438.592477083206, 'train_loss': 0.05953135157112407, 'val_loss': 0.1697242379425064}
        ]

        return{
            "default_code_same_dim": default_code_same_dim,
            "x_update_alpha": x_update_alpha,
            "x_update_alpha_opt": x_update_alpha_opt        
            }


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

    def calculate_average_metrics(self, data_list):
        """
        Computes and returns a dataframe with the average values for each metric.
        """
        if not data_list:
            return None
        
        df = pd.DataFrame(data_list)

        # Select only numerical columns
        numeric_columns = self.metrics
        
        # Compute mean values
        df_mean = df[numeric_columns].mean().to_frame(name="Average").reset_index()
        df_mean.rename(columns={"index": "Metric"}, inplace=True)

        return df_mean


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

    def find_best_run(self, data_list):
        mode = "sort"

        if      mode == "fun":      return self.find_best_run___fun(data_list)
        elif    mode == "sort":     return self.find_best_run___sort(data_list)
        else:   return None


    def find_best_run___sort(self, data_list):
        """
        Finds the best run based on a trade-off between epoch count and loss values.
        """
        if not data_list:
            return None

        weights_1 = {
            "test_loss": 0.5,
            "train_loss": 0.2,
            "val_loss": 0.2,
            "best_epoch": 0.1  
        }

        weights_2 = {
            "test_loss": 0.6,  # Test loss is the most important.
            "train_loss": 0.1,  # Train loss is less critical.
            "val_loss": 0.2,  # Validation loss is still important.
            "best_epoch": 0.1  # Epoch count has low priority.
        }


        # Function to compute a score for each run
        weights = weights_1
        def score(run):
            return (
                run["test_loss"] * weights["test_loss"] +
                run["train_loss"] * weights["train_loss"] +
                run["val_loss"] * weights["val_loss"] +
                run["best_epoch"] * weights["best_epoch"]
            )

        # Find the run with the lowest score
        best_run = min(data_list, key=score)

        return best_run


    def find_best_run___fun(self, data_list, tradeoff_function=None):
        """
        Finds the best run based on an external trade-off function.
        
        :param data_list: List of runs.
        :param tradeoff_function: A function that takes a run dictionary and returns a score.
        """
        if not data_list:
            return None

        # Default sorting (same as before)
        if tradeoff_function is None:
            tradeoff_function__1 = lambda x: (x["test_loss"], x["train_loss"], x["val_loss"], x["best_epoch"])
            tradeoff_function__2 = lambda run: (run["test_loss"] * 0.6 + run["val_loss"] * 0.3 + run["train_loss"] * 0.1) * (1 + run["best_epoch"] / 100)

        # Find the best run based on the custom function
        best_run = min(data_list, key=tradeoff_function__2)

        return best_run

    def my_tradeoff(run):
        return (run["test_loss"] * 0.6 + run["val_loss"] * 0.3 + run["train_loss"] * 0.1) * (1 + run["best_epoch"] / 100)



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

    def display_results(self, dataset_name, data_list):
        if not data_list:
            print("No data available")
            return

        df = pd.DataFrame(data_list)

        numeric_columns = self.metrics

        df_mean = df[numeric_columns].mean().to_frame(name="Average").reset_index()
        df_mean.rename(columns={"index": "Metric"}, inplace=True)

        # Find best runs
        best_run___sort = self.find_best_run___sort(data_list)
        best_run___fun = self.find_best_run___fun(data_list)

        if best_run___sort is None or best_run___fun is None:
            print("Error: No best run found.")
            return

        # Convert best runs to DataFrame (Ensure to keep only relevant columns)
        best_run1_df = pd.DataFrame.from_dict(best_run___sort, orient='index', columns=["Best Run (sort)"]).reset_index()
        best_run1_df.rename(columns={"index": "Metric"}, inplace=True)

        best_run2_df = pd.DataFrame.from_dict(best_run___fun, orient='index', columns=["Best Run (fun)"]).reset_index()
        best_run2_df.rename(columns={"index": "Metric"}, inplace=True)

        # Remove unwanted columns (e.g., '_runtime', '_wandb')
        best_run1_df = best_run1_df[best_run1_df["Metric"].isin(numeric_columns)]
        best_run2_df = best_run2_df[best_run2_df["Metric"].isin(numeric_columns)]

        # Merge data while keeping a single "Metric" column
        merged_df = pd.merge(df_mean, best_run1_df, on="Metric", how="left")
        merged_df = pd.merge(merged_df, best_run2_df, on="Metric", how="left")

        # Add Run Name Row
        run_name_row = pd.DataFrame({"Metric": ["Run Name"], 
                                    "Average": [""], 
                                    "Best Run (sort)": [best_run___sort.get('run', 'N/A')], 
                                    "Best Run (fun)": [best_run___fun.get('run', 'N/A')]})

        # Concatenate final dataframe
        merged_df = pd.concat([run_name_row, merged_df], ignore_index=True)

        # Format numbers to 6 decimal places but remove unnecessary trailing zeros
        def format_number(x):
            if isinstance(x, (int, float)):
                return f"{x:.6f}".rstrip('0').rstrip('.')  # Removes trailing zeros
            return x  # Keep non-numeric values as they are

        merged_df.iloc[:, 1:] = merged_df.iloc[:, 1:].applymap(format_number)

        # Check column count before plotting
        if merged_df.shape[1] != 4:
            print("Error: Merged DataFrame does not have 4 columns. Found:", merged_df.shape[1])
            print(merged_df.head())
            return

        # Plot Table
        fig, ax = plt.subplots(figsize=(8, 5))  # Make figure slightly larger
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(cellText=merged_df.values, 
                        colLabels=["Metric", "Average", "Best Run (sort)", "Best Run (fun)"], 
                        cellLoc="center", loc="center")

        # Apply styling
        for (i, key), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(color="red", fontweight="bold")  # Make headers red
            elif key == 0:  # First column
                cell.set_text_props(color="blue", fontweight="bold")  # Make first column blue

        plt.title(f"Average Metrics for {dataset_name}", fontsize=12, fontweight="bold")
        plt.show()


    def display_final_comparison__(self):
        """Displays a final table comparing the best models across all datasets."""
        if not self.datasets:
            print("No datasets available.")
            return

        numeric_columns = self.metrics

        # Find best model for each dataset
        best_models = {name: self.find_best_run(data) for name, data in self.datasets.items()}

        # Extract numeric values for each dataset
        comparison_data = {dataset: [best_models[dataset].get(metric, "N/A") for metric in numeric_columns]
                        for dataset in self.datasets}

        # Convert to DataFrame (rows = metrics, columns = datasets)
        comparison_df = pd.DataFrame(comparison_data, index=numeric_columns)

        run_names = {dataset: best_models[dataset].get("run", "N/A") for dataset in self.datasets}
        comparison_df = pd.concat([pd.DataFrame(run_names, index=["Run Name"]), comparison_df])

        # Format numbers correctly (rounding to 6 decimals)
        for col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.6f}".rstrip('0').rstrip('.') if isinstance(x, (int, float)) else x)

        # Plot Table
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(cellText=comparison_df.values,
                        colLabels=comparison_df.columns,
                        rowLabels=comparison_df.index,
                        cellLoc="center", loc="center")

        # Apply styling
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row (datasets)
                cell.set_text_props(color="red", fontweight="bold")
            elif j == -1:  # First column (metrics)
                cell.set_text_props(color="blue", fontweight="bold")

        plt.title("Comparison of Best Models Across Datasets", fontsize=12, fontweight="bold")
        plt.show()


    def display_final_comparison(self):
        """Displays a final table comparing the best models across all datasets, highlighting min values in green."""
        if not self.datasets:
            print("No datasets available.")
            return

        numeric_columns = self.metrics

        # Find best model for each dataset
        best_models = {name: self.find_best_run(data) for name, data in self.datasets.items()}

        # Extract numeric values for each dataset
        comparison_data = {dataset: [best_models[dataset].get(metric, "N/A") for metric in numeric_columns]
                        for dataset in self.datasets}

        # Convert to DataFrame (rows = metrics, columns = datasets)
        comparison_df = pd.DataFrame(comparison_data, index=numeric_columns)

        # Add "Run Name" row at the top (Do NOT include it in min search)
        run_names = {dataset: best_models[dataset].get("run", "N/A") for dataset in self.datasets}
        run_name_df = pd.DataFrame(run_names, index=["Run Name"])
        final_df = pd.concat([run_name_df, comparison_df])

        # Convert all numeric values back to floats for proper min calculations
        for col in final_df.columns:
            final_df[col] = final_df[col].apply(lambda x: float(x) if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit() else x)

        # Format numbers correctly (rounding to 6 decimals)
        for col in final_df.columns:
            final_df[col] = final_df[col].apply(lambda x: f"{x:.6f}".rstrip('0').rstrip('.') if isinstance(x, (int, float)) else x)

        # Plot Table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(cellText=final_df.values,
                        colLabels=final_df.columns,
                        rowLabels=final_df.index,
                        cellLoc="center", loc="center")

        # Apply styling
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row (datasets)
                cell.set_text_props(color="black", fontweight="bold")
            elif j == -1:  # First column (metric names)
                cell.set_text_props(color="black", fontweight="bold")

        # Identify min values and color them green (Ignore "Run Name" row)
        for i in range(1, len(final_df)):  # Start from 1 to skip "Run Name"
            numeric_values = {}
            for j, col in enumerate(final_df.columns):
                try:
                    num_value = float(final_df.iloc[i, j])  # Convert to float
                    numeric_values[j] = num_value  # Store valid numeric values
                except ValueError:
                    continue  # Ignore non-numeric values

            if numeric_values:  # Ensure there's at least one valid number
                min_value = min(numeric_values.values())  # Find the min value
                max_value = max(numeric_values.values())

                for j, value in numeric_values.items():
                    if value == min_value:
                        table[i+1, j].set_text_props(color="green", fontweight="bold")  # Highlight in green
                    if value == max_value: 
                        table[i+1, j].set_text_props(color="red", fontweight="bold")

        plt.title("Comparison of Best Models Across Datasets", fontsize=12, fontweight="bold")
        plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

CalcValues()