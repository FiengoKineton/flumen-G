"""import wandb
run = wandb.init()
artifact = run.use_artifact('aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/model_checkpoint:v53', type='model')
artifact_dir = artifact.download()"""


import pandas as pd
import ast

def filter_top_n_by_metric(file_path, n, id_ranges, metric, best=True):
    # Read CSV file
    df = pd.read_csv(file_path, index_col=0)
    
    # Convert summary column from string to dictionary
    df['summary'] = df['summary'].apply(ast.literal_eval)
    
    # Extract val_loss, test_loss, and train_loss
    df['val_loss'] = df['summary'].apply(lambda x: x.get('val_loss', float('inf')))
    df['test_loss'] = df['summary'].apply(lambda x: x.get('test_loss', float('inf')))
    df['train_loss'] = df['summary'].apply(lambda x: x.get('train_loss', float('inf')))
    
    # Extract ID number and filter based on given ID ranges
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID_number'}, inplace=True)
    
    valid_ids = set()
    for r in id_ranges:
        if isinstance(r, tuple):
            valid_ids.update(set(range(r[0], r[1] + 1)))
        else:
            valid_ids.add(r)
    
    df = df[df['ID_number'].isin(valid_ids)]
    
    # Select top n by lowest val_loss
    df = df.nsmallest(n, metric) if best else df.nlargest(n, metric)
    
    # Format output
    df_output = df[['name', 'val_loss', 'test_loss', 'train_loss', 'ID_number']]
    df_output.columns = ['', 'val_loss', 'test_loss', 'train_loss', '']
    
    # Print table
    print(f"\n\nTop {n} results by {metric}:")
    print(df_output.to_string(index=False))
    
    return df_output

# Example usage
file_path = "run_data/wandb_get_runs.csv"  # Use the correct relative path
starting_point = 15
id_ranges = [(starting_point,51)]  # Define ID ranges as [(start1, end1), (start2, end2), ...]

"""best = False
print("\nOrder by max:")
filter_top_n_by_metric(file_path, 5, id_ranges, 'val_loss', best)
filter_top_n_by_metric(file_path, 5, id_ranges, 'test_loss', best)
filter_top_n_by_metric(file_path, 5, id_ranges, 'train_loss', best)"""

best = True
print("\n---------------------------------------\nOrder by min:\n---------------------------------------")
filter_top_n_by_metric(file_path, 5, id_ranges, 'val_loss', best)
filter_top_n_by_metric(file_path, 5, id_ranges, 'test_loss', best)
filter_top_n_by_metric(file_path, 5, id_ranges, 'train_loss', best)
