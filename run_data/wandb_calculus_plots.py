import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


class DataVisualizer:
    def __init__(self, csv_file, all=False):
        """
        Initialize the DataVisualizer class by loading the CSV file
        and extracting relevant data.
        """
        self.data = pd.read_csv(csv_file)
        
        # Convert summary and config columns from string to dictionary
        self.data['summary'] = self.data['summary'].apply(ast.literal_eval)
        self.data['config'] = self.data['config'].apply(ast.literal_eval)

        # Extract relevant metrics
        self.metrics = ['val_loss', 'test_loss', 'train_loss', 'time']
        for metric in self.metrics:
            self.data[metric] = self.data['summary'].apply(lambda x: x.get(metric, float('inf')))

        # Extract hyperparameters for analysis
        self.params = [
            'control_rnn_size', 'encoder_size', 'encoder_depth', 'decoder_size',
            'decoder_depth', 'batch_size', 'lr', 'loss', 'optimiser_mode',
            'discretisation_mode', 'x_update_mode'
        ]
        for param in self.params:
            self.data[param] = self.data['config'].apply(lambda x: x.get(param, None))

        # Set seaborn style
        sns.set_style("whitegrid")

        if all: 
            self.plot_box()
            self.plot_distribution()
            self.plot_trend()
            self.plot_pairplot()
            self.plot_correlation_heatmap()
            self.plot_val_loss_vs_hyperparams()
        else:
            #self.plot_box()
            #self.plot_distribution()
            #self.plot_trend()
            #self.plot_pairplot()
            #self.plot_correlation_heatmap()
            self.plot_val_loss_vs_hyperparams()

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Run results analysis with optional display and plotting.")
        parser.add_argument("--all", action="store_true", help="Select all metrics across datasets.")
        args = parser.parse_args()
        return args


    def plot_box(self):
        """
        Box Plot of Each Metric
        Displays the distribution of each metric. 
        The box represents the interquartile range (middle 50% of the data), 
        with the median inside the box and outliers displayed as individual points.
        Log scale is applied for better visualization.
        """
        for metric in self.metrics:
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=self.data[metric])
            plt.title(f'Box Plot of {metric}')
            plt.ylabel(metric)
            plt.yscale('log')  # Log scale for better visualization
            plt.show()

    def plot_distribution(self):
        """
        Distribution Plot of Each Metric
        Histogram visualization of each metric’s distribution.
        Kernel Density Estimation (KDE) curve added to illustrate the probability distribution.
        Log scale is applied to better accommodate outliers.
        """
        for metric in self.metrics:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[metric], bins=20, kde=True)
            plt.title(f'Distribution of {metric}')
            plt.xlabel(metric)
            plt.ylabel('Frequency')
            plt.yscale('log')
            plt.show()

    def plot_trend(self):
        """
        Trend of Metrics Over Runs
        Line plot showing how each metric evolves across training runs.
        Helps to observe trends, such as whether a loss metric decreases over time.
        Log scale is applied for better readability.
        """
        plt.figure(figsize=(12, 6))
        for metric in self.metrics:
            plt.plot(self.data.index, self.data[metric], marker='o', label=metric)
        plt.title('Trend of Metrics Over Runs')
        plt.xlabel('Run Index')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.yscale('log')
        plt.show()

    def plot_pairplot(self):
        """
        Pairplot of Metrics
        A matrix of scatter plots showing relationships between different metrics.
        If a trend is visible, it indicates a correlation between metrics.
        """
        sns.pairplot(self.data[self.metrics])
        plt.show()

    def plot_correlation_heatmap(self):
        """
        Correlation Heatmap of Metrics
        Heatmap showing correlation coefficients between different metrics.
        Values near 1 indicate strong positive correlation, while negative values indicate inverse relationships.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.data[self.metrics].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of Metrics')
        plt.show()

    def plot_val_loss_vs_hyperparams(self):
        """
        Val Loss vs Hyperparameters
        Scatter plots showing how validation loss changes based on different hyperparameter values.
        Helps determine which hyperparameters significantly impact performance.
        """
        for param in self.params:
            category_mapping = None  # Initialize mapping

            # Check if the column is categorical (string values)
            if self.data[param].dtype == 'object':
                self.data[param] = self.data[param].astype('category')  # Convert to categorical
                category_mapping = dict(enumerate(self.data[param].cat.categories))  # Store category mapping
                self.data[param] = self.data[param].cat.codes  # Convert to numeric codes
            
            # Ensure parameter is numerical and has no missing values
            if pd.api.types.is_numeric_dtype(self.data[param]):
                filtered_data = self.data.dropna(subset=[param, 'val_loss'])

                if not filtered_data.empty:
                    plt.figure(figsize=(12, 6))
                    
                    if category_mapping:
                        # Use stripplot for better spacing of categorical values
                        sns.stripplot(x=filtered_data[param], y=filtered_data['val_loss'], jitter=True)
                        plt.xticks(ticks=list(category_mapping.keys()), labels=list(category_mapping.values()), rotation=45)
                    else:
                        sns.scatterplot(x=filtered_data[param], y=filtered_data['val_loss'])

                    plt.title(f'val_loss vs {param}')
                    plt.xlabel(param)
                    plt.ylabel('val_loss')
                    plt.yscale('log')
                    plt.show()




if __name__ == "__main__":
    loc_1 = "run_data/wandb_get_runs.csv"
    loc_2 = "run_data/temp.csv"
    loc_3 = "run_data/same_hyperparams.csv"

    csv_file = loc_1
    args = DataVisualizer.parse_arguments()
    DataVisualizer(csv_file, all=args.all)