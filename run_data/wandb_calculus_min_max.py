import pandas as pd
import ast
import argparse

dir = "run_data/csv_files/wandb_get_runs.csv"

table = [
    'mode_rnn', 
    'mode_dnn', 
    'discretisation_mode', 
    'x_update_mode', 
    'optimiser_mode',
    'control_rnn_size', 
    'batch_size', 
    'lr',
    'loss',
    ]


class FilterParam:
    def __init__(self, file_path, metrics, which, low_thresh=0.05, high_thresh=0.15):
        self.file_path = file_path
        self.metrics = metrics
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.which = which
        self.run()

    def run(self):
        df = pd.read_csv(self.file_path, skiprows=range(1, 15))  # Skip first 14 rows, keep header

        # Convert string to dict
        df['summary'] = df['summary'].apply(ast.literal_eval)
        df['config'] = df['config'].apply(ast.literal_eval)

        # Extract val_loss
        df[self.which] = df['summary'].apply(lambda x: x.get(self.which, float('inf')))

        # Extract config metrics
        for m in self.metrics:
            df[m] = df['config'].apply(lambda x: x.get(m, 'N/A'))

        # Define columns for output
        columns = ['name', self.which] + self.metrics

        # Filter and sort
        df_low = df[df[self.which] < self.low_thresh].sort_values(by=self.which)
        df_high = df[(df[self.which] > self.high_thresh) & (df[self.which] < 1)].sort_values(by=self.which)

        # Output
        print("\n--- Runs with", self.which, "< {} ---\n".format(self.low_thresh))
        print(df_low[columns].to_string(index=False))

        print("\n\n--- Runs with", self.which, "> {} ---\n".format(self.high_thresh))
        print(df_high[columns].to_string(index=False))


    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Run results analysis with optional display and plotting.")
        parser.add_argument("--loc", type=str, help="Path to the directory")
        parser.add_argument("--which", type=str, help="Filtered param")
        args = parser.parse_args()
        return args
    


if __name__ == "__main__":
    args = FilterParam.parse_arguments()

    file_path = dir if args.loc is None else args.loc
    which = 'val_loss' if args.which is None else args.which
    metrics = table


    FilterParam(file_path=file_path, metrics=metrics, which=which)
