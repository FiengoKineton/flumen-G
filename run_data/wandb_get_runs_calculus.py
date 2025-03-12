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
n = 5
starting_point = 15
end_point = 54
id_ranges = [(starting_point, end_point)]  # Define ID ranges as [(start1, end1), (start2, end2), ...]

"""best = False
print("\nOrder by max:")
filter_top_n_by_metric(file_path, n, id_ranges, 'val_loss', best)
filter_top_n_by_metric(file_path, n, id_ranges, 'test_loss', best)
filter_top_n_by_metric(file_path, n, id_ranges, 'train_loss', best)"""

best = True
print("\n---------------------------------------\nOrder by min:\n---------------------------------------")
filter_top_n_by_metric(file_path, n, id_ranges, 'val_loss', best)
filter_top_n_by_metric(file_path, n, id_ranges, 'test_loss', best)
filter_top_n_by_metric(file_path, n, id_ranges, 'train_loss', best)



"""
radiant-sweep-4 for val_loss
[
    46

    , "{'_runtime': 24134.2796855, '_step': 89, '_timestamp': 1741783641.2791045, '_wandb': {'runtime': 24134}, 
        'batch_size': 128, 'best_epoch': 69, 'best_test': 0.11909919634224876, 'best_train': 0.012003492137722713, 
        'best_val': 0.019937452165380357, 'epoch': 89, 'lr': 0.000125, 'n_epochs': 500, 'test_loss': 0.11996189498948671, 
        'time': 24096.859586954117, 'train_loss': 0.01148234441304806, 'val_loss': 0.021571947232125296}"
        
    , "{'lr': 0.0005, 'loss': 'mse', 'es_delta': 1e-07, 'n_epochs': 500, 'batch_size': 128, 'es_patience': 20, 
        'decoder_size': 2, 'encoder_size': 1, 'sched_factor': 2, 'decoder_depth': 2, 'encoder_depth': 1, 
        'x_update_mode': 'alpha', 'optimiser_mode': 'adam', 'sched_patience': 10, 'control_rnn_size': 20, 
        'control_rnn_depth': 1, 'discretisation_mode': 'FE'}"

    , radiant-sweep-4
]
"""
