import pandas as pd
import matplotlib.pyplot as plt


class CalcValues:
    def __init__(self):
        self.datasets = self.DataSet()
        
        """# Loop through all datasets and calculate their average metrics
        for name, data in self.datasets.items():
            print(f"\n--- Average Metrics for {name} ---\n")
            avg_metrics = self.calculate_average_metrics__(data)
            print(avg_metrics)"""

        self.average_results = {name: self.calculate_average_metrics(data) for name, data in self.datasets.items()}
        self.generate_image()


    def DataSet(self):
        return{
            "default_code_same_dim": [
                {'run': "017", '_runtime': 7546.45702, '_step': 117, '_timestamp': 1741182593.7156532, '_wandb': {'runtime': 7546}, 'best_epoch': 98, 'best_test': 0.03409814938075013, 'best_train': 0.018149984931504284, 'best_val': 0.028380416333675385, 'epoch': 118, 'lr': 0.000125, 'test_loss': 0.03472252781428988, 'time': 7528.629385232925, 'train_loss': 0.015891212812334143, 'val_loss': 0.03212327557423758},
                {'run': "026", '_runtime': 7346.7665658, '_step': 106, '_timestamp': 1741430459.320042, '_wandb': {'runtime': 7346}, 'best_epoch': 87, 'best_test': 0.025843684265892657, 'best_train': 0.014283652101993245, 'best_val': 0.02398856176388642, 'epoch': 107, 'lr': 0.000125, 'test_loss': 0.026321270325708957, 'time': 7328.914924144745, 'train_loss': 0.014671892942811446, 'val_loss': 0.026048427477242456},
                {'run': "027", '_runtime': 8290.5964382, '_step': 126, '_timestamp': 1741431693.8628545, '_wandb': {'runtime': 8290}, 'best_epoch': 107, 'best_test': 0.027500419862686643, 'best_train': 0.01666778990772193, 'best_val': 0.05470035585855681, 'epoch': 127, 'lr': 0.00025, 'test_loss': 0.027228796898963903, 'time': 8272.850373268127, 'train_loss': 0.014251226118258223, 'val_loss': 0.06482056906772038}
            ],

            "x_update_alpha": [
                {'run': "018(1)", '_runtime': 12845.8606273, '_step': 84, '_timestamp': 1741201116.3583372, '_wandb': {'runtime': 12845}, 'best_epoch': 80, 'best_test': 0.09021028982741491, 'best_train': 0.020383733487318433, 'best_val': 0.055468578689864705, 'epoch': 85, 'lr': 0.001, 'test_loss': 0.11940238163584754, 'time': 12816.142216682434, 'train_loss': 0.02383146670563196, 'val_loss': 0.0593999158591032},
                {'run': "018(2-3)", '_runtime': 29190.6178727, '_step': 156, '_timestamp': 1741217461.1155827, '_wandb': {'runtime': 29190}, 'best_epoch': 137, 'best_test': 0.08476408736573325, 'best_train': 0.01709923068327563, 'best_val': 0.04871034687237134, 'epoch': 157, 'lr': 0.000125, 'test_loss': 0.08193321597008478, 'time': 29160.899462223053, 'train_loss': 0.01672123080838925, 'val_loss': 0.04979510998560323},
                {'run': "019", '_runtime': 23343.4636111, '_step': 142, '_timestamp': 1741280362.9845507, '_wandb': {'runtime': 23343}, 'best_epoch': 123, 'best_test': 0.0533087533558645, 'best_train': 0.02122072688249684, 'best_val': 0.25057389338811237, 'epoch': 143, 'lr': 6.25e-05, 'test_loss': 0.05645283382563364, 'time': 23315.39107489586, 'train_loss': 0.01923207267034779, 'val_loss': 0.25213538606961566},
                {'run': "020", '_runtime': 19282.2622103, '_step': 109, '_timestamp': 1741293582.378608, '_wandb': {'runtime': 19282}, 'best_epoch': 90, 'best_test': 0.1449957171839381, 'best_train': 0.02281889265708665, 'best_val': 0.03265471753501703, 'epoch': 110, 'lr': 0.000125, 'test_loss': 0.14242024950328327, 'time': 19247.294664382935, 'train_loss': 0.019359053211119123, 'val_loss': 0.03382213047099492},
                {'run': "021", '_runtime': 24220.1446849, '_step': 145, '_timestamp': 1741321179.491952, '_wandb': {'runtime': 24220}, 'best_epoch': 126, 'best_test': 0.06649408028239295, 'best_train': 0.018753742099439027, 'best_val': 0.11177096456762343, 'epoch': 146, 'lr': 3.125e-05, 'test_loss': 0.06538293534328067, 'time': 24190.640423059464, 'train_loss': 0.017817277136066602, 'val_loss': 0.1155335587405023}
            ],

            "x_update_alpha_opt": [
                {'run': "022", '_runtime': 15588.0104955, '_step': 79, '_timestamp': 1741373321.7143245, '_wandb': {'runtime': 15588}, 'best_epoch': 60, 'best_test': 0.19954035431146624, 'best_train': 0.033677110980663984, 'best_val': 0.1268843865347287, 'epoch': 80, 'lr': 0.0005, 'test_loss': 0.17315763891452834, 'time': 15557.825784683228, 'train_loss': 0.02871229724771289, 'val_loss': 0.13715681126193394}, 
                {'run': "023", '_runtime': 19091.3150562, '_step': 90, '_timestamp': 1741383918.291978, '_wandb': {'runtime': 19091}, 'best_epoch': 71, 'best_test': 0.17176606902290906, 'best_train': 0.03029533599813779, 'best_val': 0.0722720061857549, 'epoch': 91, 'lr': 0.00025, 'test_loss': 0.15187193074869731, 'time': 19058.01224398613, 'train_loss': 0.021802850865892003, 'val_loss': 0.08129255662834833},
                {'run': "024", '_runtime': 16780.6603587, '_step': 74, '_timestamp': 1741391706.1888623, '_wandb': {'runtime': 16780}, 'best_epoch': 55, 'best_test': 0.29205070069384953, 'best_train': 0.030286846978087274, 'best_val': 0.058476886756363367, 'epoch': 75, 'lr': 0.00025, 'test_loss': 0.20884731518370764, 'time': 16748.41609811783, 'train_loss': 0.024770572751997007, 'val_loss': 0.06078948893599094},
                {'run': "025", '_runtime': 6466.9516759, '_step': 47, '_timestamp': 1741404462.7291248, '_wandb': {'runtime': 6466}, 'best_epoch': 28, 'best_test': 0.2853936068122349, 'best_train': 0.11861003164655318, 'best_val': 0.1437250425418218, 'epoch': 48, 'lr': 0.0005, 'test_loss': 0.3043967697118956, 'time': 6438.592477083206, 'train_loss': 0.05953135157112407, 'val_loss': 0.1697242379425064}
            ]
        }

    def calculate_average_metrics__(self, data_list):
        """
        Computes and returns a dictionary with the average values for each metric.
        """
        if not data_list:
            print("Error: The data list is empty.")
            return None
        
        df = pd.DataFrame(data_list)

        # Select only relevant numerical columns
        numeric_columns = ["best_epoch", "best_test", "best_train", "best_val", "lr", 
                           "test_loss", "time", "train_loss", "val_loss"]
        
        # Ensure required columns exist
        missing_columns = [col for col in numeric_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing columns - {missing_columns}")
            return None

        # Compute mean values
        df_mean = df[numeric_columns].mean().to_frame(name="Average").reset_index()
        df_mean.rename(columns={"index": "Metric"}, inplace=True)

        return df_mean
    

    def calculate_average_metrics(self, data_list):
        """
        Computes and returns a dataframe with the average values for each metric.
        """
        if not data_list:
            return None
        
        df = pd.DataFrame(data_list)

        # Select only numerical columns
        numeric_columns = ["best_epoch", "best_test", "best_train", "best_val", "lr", 
                           "test_loss", "time", "train_loss", "val_loss"]
        
        # Compute mean values
        df_mean = df[numeric_columns].mean().to_frame(name="Average").reset_index()
        df_mean.rename(columns={"index": "Metric"}, inplace=True)

        return df_mean
    
    def generate_image(self):
        """
        Generates an image displaying the average metrics tables.
        """
        num_tables = len(self.average_results)
        fig, axes = plt.subplots(num_tables, 1, figsize=(10, 4 * num_tables))

        if num_tables == 1:
            axes = [axes]  # Ensure axes is always iterable

        for ax, (name, df) in zip(axes, self.average_results.items()):
            if df is not None:
                ax.axis('tight')
                ax.axis('off')
                table = ax.table(cellText=df.values, colLabels=df.columns, 
                                 rowLabels=df["Metric"], loc="center", cellLoc="center")

                ax.set_title(f"Average Metrics for {name}", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig("average_metrics.png", dpi=300)
        print("Saved average metrics image as 'average_metrics.png'")



CalcValues()