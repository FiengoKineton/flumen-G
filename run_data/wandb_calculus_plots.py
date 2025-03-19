import pandas as pd
import ast
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from mpl_toolkits.mplot3d import Axes3D
import pprint


param_combinations_1 = [
    ('encoder_size', 'lr', 'decoder_size'),
    ('encoder_size', 'lr', 'decoder_depth'),
    ('batch_size', 'lr', 'decoder_size'),
    ('encoder_size', 'batch_size', 'decoder_size'),
    ('encoder_size', 'batch_size', 'decoder_depth'),
    ('decoder_size', 'lr', 'control_rnn_size'),
    ('batch_size', 'lr', 'decoder_depth'),
    ('encoder_size', 'decoder_size', 'control_rnn_size'),
    ('lr', 'decoder_size', 'encoder_depth'),
    ('batch_size', 'decoder_size', 'encoder_depth')
]

param_combinations_2 = [
    ('encoder_size', 'lr', 'decoder_size'),  # Significant influence of encoder_size, lr, and decoder_size
    ('batch_size', 'lr', 'decoder_size'),  # Important relationship with batch_size and decoder_size
    ('encoder_size', 'loss', 'optimiser_mode'),  # Influence of encoder_size, loss type, and optimizer mode
    ('decoder_size', 'lr', 'discretisation_mode'),  # Impact of decoder_size, lr, and discretisation_mode
    ('batch_size', 'loss', 'lr')  # Combined effect of batch_size, loss function, and lr
]

param_2D = {
    "param1": 'lr', 
    "param2": 'batch_size'
}

param_combinations = param_combinations_2


class DataVisualizer:
    def __init__(self, csv_file, all=False):
        """
        Initialize the DataVisualizer class by loading the CSV file
        and extracting relevant data.
        """
        self.data = pd.read_csv(csv_file)
        print(self.data.shape[0])
        
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

        self.data = self.data.replace([float('inf'), -float('inf')], float('nan'))
        self.data = self.data.dropna()

        # Set seaborn style
        sns.set_style("whitegrid")

        if all: 
            self.plot_box()
            self.plot_distribution()
            self.plot_trend()
            self.plot_pairplot()
            self.plot_correlation_heatmap()
            self.plot_val_loss_vs_hyperparams()
            self.plot_val_loss_2D()
            self.plot_val_loss_3D()
        else:
            self.plot_box()
            #self.plot_distribution()
            #self.plot_trend()
            #self.plot_pairplot()
            #self.plot_correlation_heatmap()
            #self.plot_val_loss_vs_hyperparams()
            #self.plot_val_loss_2D()
            self.plot_val_loss_3D()

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
        num_metrics = len(self.metrics)
        cols = 2
        rows = (num_metrics // cols) + (num_metrics % cols != 0)  # Determine number of rows based on metrics

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Create a grid of subplots
        axes = axes.flatten()  # Flatten the axes for easier iteration

        for i, metric in enumerate(self.metrics):
            sns.boxplot(y=self.data[metric], ax=axes[i])  # Plot on the i-th subplot
            axes[i].set_title(f'Box Plot of {metric}')
            axes[i].set_ylabel(metric)
            axes[i].set_yscale('log')  # Log scale for better visualization

        # Remove any unused subplots (in case of uneven number of metrics)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()  # Ensure no overlap between subplots
        plt.show()

    def plot_distribution(self):
        """
        Distribution Plot of Each Metric
        Histogram visualization of each metric’s distribution.
        Kernel Density Estimation (KDE) curve added to illustrate the probability distribution.
        Log scale is applied to better accommodate outliers.
        """
        num_metrics = len(self.metrics)
        cols = 2  # Number of columns in the grid
        rows = (num_metrics // cols) + (num_metrics % cols != 0)  # Determine number of rows based on metrics

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Create a grid of subplots
        axes = axes.flatten()  # Flatten the axes for easier iteration

        for i, metric in enumerate(self.metrics):
            sns.histplot(self.data[metric], bins=20, kde=True, ax=axes[i])  # Plot on the i-th subplot
            axes[i].set_title(f'Distribution of {metric}')
            axes[i].set_xlabel(metric)
            axes[i].set_ylabel('Frequency')
            axes[i].set_yscale('log')  # Apply log scale for better visualization

        # Remove any unused subplots (in case of uneven number of metrics)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()  # Ensure no overlap between subplots
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
        #"""

        # Initialize the figure and axes for subplots
        num_plots = len(self.params)
        cols = 4  # Number of columns in the grid
        rows = (num_plots // cols) + (num_plots % cols != 0)  # Determine number of rows based on params
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # Flatten the axes for easier iteration in case of multiple rows
        axes = axes.flatten()

        for i, param in enumerate(self.params):
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
                    ax = axes[i]  # Use a specific subplot for each param
                    #ax.set_title(f'val_loss vs {param}')
                    ax.set_xlabel(param)
                    ax.set_ylabel('val_loss')
                    ax.set_yscale('log')

                    if category_mapping:
                        # Use stripplot for better spacing of categorical values
                        sns.stripplot(x=filtered_data[param], y=filtered_data['val_loss'], jitter=True, ax=ax)
                        ax.set_xticks(list(category_mapping.keys()))
                        ax.set_xticklabels(list(category_mapping.values()), rotation=45)
                    else:
                        sns.scatterplot(x=filtered_data[param], y=filtered_data['val_loss'], ax=ax)

        # Remove any unused subplots (in case of uneven number of params)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()  # Ensure there's no overlap between subplots
        plt.show()

    def plot_val_loss_2D(self):
        """
        Val Loss vs Two Hyperparameters
        Scatter plot showing how the validation loss changes based on the combination of two hyperparameters.
        This helps to understand the joint effect of two hyperparameters on the performance.
        """

        param1, param2 = param_2D["param1"], param_2D["param2"]
        # Ensure both parameters are numerical and not missing
        filtered_data = self.data.dropna(subset=[param1, param2, 'val_loss'])
        filtered_data = filtered_data[(filtered_data[param1] != float('inf')) & (filtered_data[param2] != float('inf'))]

        if not filtered_data.empty:
            plt.figure(figsize=(12, 6))
            scatter = sns.scatterplot(data=filtered_data, x=param1, y=param2, hue='val_loss', palette='coolwarm', size='val_loss', sizes=(20, 200), marker='o')
            plt.title(f'Validation Loss vs {param1} and {param2}')
            plt.xlabel(param1)
            plt.ylabel(param2)
            plt.legend(title='val_loss', loc='best', labels=[f'{x}' for x in filtered_data['val_loss']])
            plt.yscale('log')  # Apply log scale to the y-axis for better visibility of loss
        else:
            print(f"No valid data found for the parameters: {param1}, {param2}")

    def plot_val_loss_3D(self):
        """
        Plot 3D scatter plots for multiple sets of parameter combinations.
        
        Args:
        - data: The dataset containing the parameters and val_loss.
        """
        # Loop through param_combinations and plot 3D scatter plots
        for i, (param1, param2, param3) in enumerate(param_combinations):
            # Check if the parameters exist in the data
            if param1 not in self.data.columns:
                print(f"Skipping {param1} is missing from the data.")
                continue
            if param2 not in self.data.columns:
                print(f"Skipping {param2} is missing from the data.")
                continue
            if param3 not in self.data.columns:
                print(f"Skipping {param3} is missing from the data.")
                continue

            # Filter out missing values from the dataset
            filtered_data = self.data.dropna(subset=[param1, param2, param3, 'val_loss'])
            old_data = filtered_data

            #pprint.pprint(filtered_data)

            if not filtered_data.empty:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Check if parameters are numeric
                if pd.api.types.is_numeric_dtype(filtered_data[param1]) and pd.api.types.is_numeric_dtype(filtered_data[param2]) and pd.api.types.is_numeric_dtype(filtered_data[param3]):
                    # 3D scatter plot for numeric parameters
                    sc = ax.scatter(filtered_data[param1], filtered_data[param2], filtered_data[param3], c=filtered_data['val_loss'], cmap='coolwarm')
                    d1, d2, d3 = None, None, None
                else:
                    # Convert non-numeric parameters to categories (use category codes)
                    print("\n\n-----------------------------------")
                    print(f"Plotting: {param1}, {param2}, {param3}")
                    print("-----------------------------------\n")

                    if not pd.api.types.is_numeric_dtype(filtered_data[param1]): 
                        c1 = filtered_data[param1].astype('category')
                        d1 = dict(enumerate(c1.cat.categories))
                        print (param1, "\n", d1)
                        filtered_data[param1] = filtered_data[param1].astype('category').cat.codes
                    else: d1 = None
                    
                    if not pd.api.types.is_numeric_dtype(filtered_data[param2]): 
                        c2 = filtered_data[param2].astype('category')
                        d2 = dict(enumerate(c2.cat.categories))
                        print (param2, "\n", d2)
                        filtered_data[param2] = filtered_data[param2].astype('category').cat.codes
                    else: d2 = None
                    
                    if not pd.api.types.is_numeric_dtype(filtered_data[param3]): 
                        c3 = filtered_data[param3].astype('category')
                        d3 = dict(enumerate(c3.cat.categories))
                        print (param3, "\n", d3)
                        filtered_data[param3] = filtered_data[param3].astype('category').cat.codes
                    else: d3 = None

                    # 3D scatter plot for categorical parameters
                    sc = ax.scatter(filtered_data[param1], filtered_data[param2], filtered_data[param3], c=filtered_data['val_loss'], cmap='coolwarm')

                ax.set_xlabel(param1)
                ax.set_ylabel(param2)
                ax.set_zlabel(param3)
                ax.set_title(f'{param1} vs {param2} vs {param3}')

                # Add color bar
                fig.colorbar(sc, ax=ax, label='val_loss')

                # Create the legend labels only if d1, d2, or d3 exist
                legend_labels = []

                if d1:
                    legend_labels.append(f"{param1} = {d1}")
                if d2:
                    legend_labels.append(f"{param2} = {d2}")
                if d3:
                    legend_labels.append(f"{param3} = {d3}")
                
                print(legend_labels)

                # Add the legend only if there are any labels to display
                if legend_labels:
                    ax.legend(legend_labels, title='Parameter Labels', loc='upper left', fontsize=10, bbox_to_anchor=(0, 1))


                # Display values near each point
                for j in range(len(filtered_data)):
                    ax.text(filtered_data[param1].iloc[j], 
                            filtered_data[param2].iloc[j], 
                            filtered_data[param3].iloc[j], 
                            f'{old_data[param1].iloc[j]}, {old_data[param2].iloc[j]}, {old_data[param3].iloc[j]}',
                            color='black', fontsize=8)
                
                plt.show()  # Display plot and wait for user to close it before the next plot
            else:
                print(f"No valid data found for the parameters: {param1}, {param2}, {param3}")



if __name__ == "__main__":
    loc_1 = "run_data/csv_files/wandb_get_runs.csv"
    loc_2 = "run_data/csv_files/temp.csv"
    loc_3 = "run_data/csv_files/sweep_test1.csv"
    loc_4 = "run_data/csv_files/sweep_test2.csv"

    csv_file = loc_4
    args = DataVisualizer.parse_arguments()
    DataVisualizer(csv_file, all=args.all)