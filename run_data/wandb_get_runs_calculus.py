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
end_point = 67
id_ranges = [(starting_point, end_point)]  # Define ID ranges as [(start1, end1), (start2, end2), ...]

"""best = False
print("\nOrder by max:")
filter_top_n_by_metric(file_path, n, id_ranges, 'val_loss', best)
filter_top_n_by_metric(file_path, n, id_ranges, 'test_loss', best)
filter_top_n_by_metric(file_path, n, id_ranges, 'train_loss', best)"""

best = True
print("\n---------------------------------------\nOrder by min:\n---------------------------------------")
name = ['val_loss'] #, 'test_loss', 'train_loss']
for name in name:   filter_top_n_by_metric(file_path, n, id_ranges, name, best)




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


---------------------------------------
Order by min:
---------------------------------------


Top 5 results by val_loss:
                             val_loss  test_loss  train_loss   
            radiant-sweep-4  0.021572   0.119962    0.011482 46
              swift-sweep-1  0.023584   0.150588    0.013035 40
026___default-code-same-dim  0.026048   0.026321    0.014672 28
       037___improving-beta  0.028786   0.054766    0.018741 39
       034___improving-beta  0.032041   0.105041    0.016758 36


Top 5 results by test_loss:
                             val_loss  test_loss  train_loss   
026___default-code-same-dim  0.026048   0.026321    0.014672 28
027___default-code-same-dim  0.064821   0.027229    0.014251 29
        033___x-update-beta  0.055233   0.028553    0.017269 35
017___default-code-same-dim  0.032123   0.034723    0.015891 19
      035___improving-alpha  0.054753   0.036374    0.017880 37


Top 5 results by train_loss:
                             val_loss  test_loss  train_loss   
    038___hyperparams-set-1  0.164580   0.040000    0.009215 55
041___hyperparams-radiant-4  0.111757   0.047922    0.009571 58
             lively-sweep-5  0.224687   0.133997    0.011310 48
            radiant-sweep-4  0.021572   0.119962    0.011482 46
          glamorous-sweep-1  0.075458   0.085277    0.012813 43
"""
