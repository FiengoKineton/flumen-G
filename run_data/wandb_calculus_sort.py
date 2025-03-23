"""import wandb
run = wandb.init()
artifact = run.use_artifact('aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/model_checkpoint:v53', type='model')
artifact_dir = artifact.download()"""


import pandas as pd
import argparse
import ast

class Sort(): 
    def __init__(self, loc=None, all=False):
        # Example usage
        file_path_1 = "run_data/csv_files/wandb_get_runs.csv"  # Use the correct relative path
        file_path_2 = "run_data/csv_files/temp.csv"  # Use the correct relative path
        file_path_3 = "run_data/csv_files/sweep_test1.csv"
        file_path_4 = "run_data/csv_files/sweep_test2.csv"
        file_path_5 = "run_data/csv_files/models.csv"
        file_path_6 = "run_data/csv_files/sweep_test3.csv"


        file_path = file_path_6 if loc is None else loc
        df = pd.read_csv(file_path)

        n = 5 if file_path == file_path_1 else df.shape[0]
        starting_point = 15 if file_path == file_path_1 else 1
        end_point = df.shape[0]
        id_ranges = [(starting_point, end_point)]

        """best = False
        print("\nOrder by max:")
        filter_top_n_by_metric(file_path, n, id_ranges, 'val_loss', best)
        filter_top_n_by_metric(file_path, n, id_ranges, 'test_loss', best)
        filter_top_n_by_metric(file_path, n, id_ranges, 'train_loss', best)"""

        best = True
        print("\n---------------------------------------\nOrder by min:\n---------------------------------------")
        name = ['val_loss', 'best_val', 'test_loss', 'train_loss', 'time'] if all else ['val_loss', 'best_val']
        for name in name:   self.filter_top_n_by_metric(file_path, n, id_ranges, name, best)


    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Run results analysis with optional display and plotting.")
        parser.add_argument("--loc", type=str, help="Path to the directory")
        parser.add_argument("--all", action="store_true", help="Select all metrics across datasets.")
        args = parser.parse_args()
        return args
    

    def filter_top_n_by_metric(self, file_path, n, id_ranges, metric, best=True):
        # Read CSV file
        df = pd.read_csv(file_path, index_col=0)
        
        # Convert summary column from string to dictionary
        df['summary'] = df['summary'].apply(ast.literal_eval)
        
        # Extract val_loss, test_loss, and train_loss
        df['val_loss'] = df['summary'].apply(lambda x: x.get('val_loss', float('inf')))
        df['best_val'] = df['summary'].apply(lambda x: x.get('best_val', float('inf')))
        df['test_loss'] = df['summary'].apply(lambda x: x.get('test_loss', float('inf')))
        df['train_loss'] = df['summary'].apply(lambda x: x.get('train_loss', float('inf')))
        df['time'] = df['summary'].apply(lambda x: x.get('time', float('inf')))
        
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
        df_output = df[['name', 'val_loss', 'best_val', 'test_loss', 'train_loss', 'time', 'ID_number']]
        df_output.columns = ['', 'val_loss', 'best_val', 'test_loss', 'train_loss', 'time', '']
        
        # Print table
        print(f"\n\nTop {n} results by {metric}:")
        print(df_output.to_string(index=False))
        
        return df_output



if __name__ == "__main__":
    args = Sort.parse_arguments()
    Sort(loc=args.loc, all=args.all) 


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



############### file_path_1 ###################################################################

---------------------------------------
Order by min:
---------------------------------------



Top 5 results by val_loss:
                             val_loss  test_loss  train_loss         time   
            radiant-sweep-4  0.021572   0.119962    0.011482 24096.859587 46
              swift-sweep-1  0.023584   0.150588    0.013035 18822.825367 40
026___default-code-same-dim  0.026048   0.026321    0.014672  7328.914924 28
       037___improving-beta  0.028786   0.054766    0.018741 23409.787544 39
  047___hyperparams___set_3  0.028925   0.032248    0.007317 15343.459737 68


Top 5 results by test_loss:
                             val_loss  test_loss  train_loss         time   
026___default-code-same-dim  0.026048   0.026321    0.014672  7328.914924 28
027___default-code-same-dim  0.064821   0.027229    0.014251  8272.850373 29
        033___x-update-beta  0.055233   0.028553    0.017269 11577.355808 35
  047___hyperparams___set_3  0.028925   0.032248    0.007317 15343.459737 68
017___default-code-same-dim  0.032123   0.034723    0.015891  7528.629385 19


Top 5 results by train_loss:
                             val_loss  test_loss  train_loss         time   
  047___hyperparams___set_3  0.028925   0.032248    0.007317 15343.459737 68
             hearty-sweep-2  0.096895   0.319395    0.007751 25397.204516 65
    038___hyperparams-set-1  0.164580   0.040000    0.009215 18786.769175 55
041___hyperparams-radiant-4  0.111757   0.047922    0.009571 23089.858831 58
  049___hyperparams___set_5  0.052925   0.289066    0.010115 11965.970435 71


Top 5 results by time:
                               val_loss  test_loss  train_loss        time
048___hyperparams___set_4(nf)  0.252035   0.089299    0.067268 1626.147287 69
          012___adam1_TU1_nf4  0.163657   0.193974    0.099135 3217.721420 15
             faithful-sweep-4  0.095243   0.107093    0.013504 4381.372042 47
            014___super-wrong  3.968048   3.987160    4.052914 4679.883023 16
         025___x-update-alpha  0.169724   0.304397    0.059531 6438.592477 27




############### file_path_2 ###################################################################


46, radiant-sweep-4,                
40, swift-sweep-1,                 
28, 026___default-code-same-dim,    
35, 033___x-update-beta,            
65, hearty-sweep-2,                 
55, 038___hyperparams-set-1,       
47, faithful-sweep-4,  
68, 047___hyperparams___set_3,



---------------------------------------
Order by min:
---------------------------------------



Top 2 results by val_loss:
                 val_loss  test_loss  train_loss         time  
radiant-sweep-4  0.021572   0.119962    0.011482 24096.859587 1
  swift-sweep-1  0.023584   0.150588    0.013035 18822.825367 2


Top 2 results by test_loss:
                             val_loss  test_loss  train_loss         time  
026___default-code-same-dim  0.026048   0.026321    0.014672  7328.914924 3
        033___x-update-beta  0.055233   0.028553    0.017269 11577.355808 4


Top 2 results by train_loss:
                         val_loss  test_loss  train_loss         time  
         hearty-sweep-2  0.096895   0.319395    0.007751 25397.204516 5
038___hyperparams-set-1  0.164580   0.040000    0.009215 18786.769175 6


Top 2 results by time:
                             val_loss  test_loss  train_loss        time  
           faithful-sweep-4  0.095243   0.107093    0.013504 4381.372042 7
026___default-code-same-dim  0.026048   0.026321    0.014672 7328.914924 3
"""
