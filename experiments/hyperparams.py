import pandas as pd
import ast
import numpy as np
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from sklearn.preprocessing import MinMaxScaler


K = 15
PRIOR_SETS = ["hyperparams___set_1", 
              "hyperparams___set_2", 
              "hyperparams___radiant_sweep_4", 
              "hyperparams___swift_sweep_1",
              "hyperparams___opt_best_1",
              "hyperparams___opt_best_2",
              "hyperparams___opt_bayes_1",
              "hyperparams___opt_bayes_2",
              ]



class Hyperparams:
    def __init__(self):
        self.hyperparams_sets = {
            'hyperparams___init': {
                'control_rnn_size': 8,          ### default 12 | try 20 | 8 if self.mode_rnn="true" else 10
                'control_rnn_depth': 1,         ### maybe try 2? (num_layer == control_rnn_depth) --- Nope!
                'encoder_size': 1,
                'encoder_depth': 2,
                'decoder_size': 1,
                'decoder_depth': 2,
                'batch_size': 128,
                'lr': 0.001,                    ### try 5e-4 to increase stability
                'n_epochs': 1000,
                'es_patience': 20,              ### default 20
                'es_delta': 1e-7,
                'sched_patience': 10,
                'sched_factor': 2,
                'loss': "mse",
                'discretisation_mode': "TU",    #-- {TU, FE, BE}
                'optimiser_mode': "adam",       #-- {adam, tbptt, nesterov, newton}
                'x_update_mode': "alpha",       #-- {alpha, beta, lamda}
            },


            'hyperparams___set_1': {
                "control_rnn_size": 20,         
                "control_rnn_depth": 1,         
                "encoder_size": 1,   
                "encoder_depth": 2,  
                "decoder_size": 2,  
                "decoder_depth": 2,  
                "batch_size": 128,  
                "lr": 0.0005,                    
                "n_epochs": 500,  
                "es_patience": 20,              
                "es_delta": 1e-07,
                "sched_patience": 10,
                "sched_factor": 2,
                "loss": "mse",
                "discretisation_mode": "FE",    
                "optimiser_mode": "adam",       
                "x_update_mode": "alpha"       
            },

            'hyperparams___set_2': {
                "control_rnn_size": 8,         
                "control_rnn_depth": 1,         
                "encoder_size": 2,   
                "encoder_depth": 1,  
                "decoder_size": 2,  
                "decoder_depth": 1,  
                "batch_size": 64,  
                "lr": 0.00075,                    
                "n_epochs": 600,  
                "es_patience": 10,              
                "es_delta": 1e-07,
                "sched_patience": 10,
                "sched_factor": 2,
                "loss": "mse",
                "discretisation_mode": "FE",    
                "optimiser_mode": "adam",       
                "x_update_mode": "alpha"       
            },


            'hyperparams___radiant_sweep_4': {'lr': 0.0005, 'loss': 'mse', 'es_delta': 1e-07, 'n_epochs': 500, 'batch_size': 128, 'es_patience': 20, 'decoder_size': 2, 'encoder_size': 1, 'sched_factor': 2, 'decoder_depth': 2, 'encoder_depth': 1, 'x_update_mode': 'alpha', 'optimiser_mode': 'adam', 'sched_patience': 10, 'control_rnn_size': 20, 'control_rnn_depth': 1, 'discretisation_mode': 'FE'},
            'hyperparams___swift_sweep_1': {'lr': 0.0005, 'loss': 'mse', 'es_delta': 1e-07, 'n_epochs': 500, 'batch_size': 256, 'es_patience': 20, 'decoder_size': 1, 'encoder_size': 2, 'sched_factor': 2, 'decoder_depth': 1, 'encoder_depth': 1, 'x_update_mode': 'beta', 'optimiser_mode': 'adam', 'sched_patience': 10, 'control_rnn_size': 12, 'control_rnn_depth': 1, 'discretisation_mode': 'TU'},
        
            # from _try_opt:
            'hyperparams___opt_best_1': {'lr': 0.001, 'loss': 'mse', 'es_delta': 1e-07, 'n_epochs': 1000, 'batch_size': 128, 'es_patience': 20, 'decoder_size': 1, 'encoder_size': 1, 'sched_factor': 2, 'decoder_depth': 2, 'encoder_depth': 2, 'x_update_mode': 'beta', 'optimiser_mode': 'adam', 'sched_patience': 10, 'control_rnn_size': 8, 'control_rnn_depth': 1, 'discretisation_mode': 'TU'},
            'hyperparams___opt_best_2': {'lr': 0.001, 'loss': 'mse', 'es_delta': 1e-07, 'n_epochs': 1000, 'batch_size': 128, 'es_patience': 20, 'decoder_size': 1, 'encoder_size': 1, 'sched_factor': 2, 'decoder_depth': 2, 'encoder_depth': 2, 'x_update_mode': 'alpha', 'optimiser_mode': 'adam', 'sched_patience': 10, 'control_rnn_size': 8, 'control_rnn_depth': 1, 'discretisation_mode': 'TU'},

            # from try_opt:
            'hyperparams___opt_bayes_1': {'lr': 0.0005, 'loss': 'mse', 'es_delta': 1e-07, 'n_epochs': 500, 'batch_size': 128, 'es_patience': 20, 'decoder_size': 2, 'encoder_size': 1, 'sched_factor': 2, 'decoder_depth': 2, 'encoder_depth': 1, 'x_update_mode': 'alpha', 'optimiser_mode': 'adam', 'sched_patience': 10, 'control_rnn_size': 20, 'control_rnn_depth': 1, 'discretisation_mode': 'FE'},
            'hyperparams___opt_bayes_2': {'lr': 0.0005, 'loss': 'mse', 'es_delta': 1e-07, 'n_epochs': 500, 'batch_size': 128, 'es_patience': 20, 'decoder_size': 2, 'encoder_size': 1, 'sched_factor': 2, 'decoder_depth': 2, 'encoder_depth': 1, 'x_update_mode': 'alpha', 'optimiser_mode': 'adam', 'sched_patience': 10, 'control_rnn_size': 20, 'control_rnn_depth': 1, 'discretisation_mode': 'FE'},
        }
    
    
    def get_hyperparams(self, name):
        return self.hyperparams_sets.get(name, f"Hyperparameter set '{name}' not found.")


# ---------------- OPTIMAL Problem | top best runs ----------------------------------------------- #

    def _opt_best(self):
        """
        Identifies the optimal hyperparameter set based on the top-performing runs.

        ## **Mathematical Formulation of the Optimization Problem**
        
        This function finds the optimal hyperparameters by selecting the **most frequently occurring** values 
        from the top \( k \) runs with the lowest validation loss. Formally, we define:

        \[
        Theta_{text{opt}} = arg\max_{theta \in Theta_k} \sum_{i=1}^{k} \mathbb{I}(theta_i = theta)
        \]

        where:

        - \( Theta_{text{opt}} \) is the set of optimized hyperparameters.
        - \( Theta_k \) is the set of hyperparameters from the top \( k \) runs with the lowest validation loss.
        - \( \mathbb{I}(\cdot) \) is an indicator function that counts occurrences of each hyperparameter.

        ## **Process Overview**
        
        1. **Extract Performance Metrics**: Load experiment results from `wandb_get_runs.csv`, focusing on:
        - \( L_{text{val}}(theta) \) (Validation loss)
        - \( L_{text{train}}(theta) \) (Training loss)
        - \( L_{text{test}}(theta) \) (Test loss)

        2. **Select Top \( k \) Runs**: Identify the \( k \) runs with the smallest validation loss:

        \[
        Theta_k = \operatorname{argmin}_{Theta} L_{text{val}}(theta), \quad k=10
        \]

        3. **Extract Hyperparameters**: Store the hyperparameter configurations for these top runs.

        4. **Compute Mode per Parameter**: For each hyperparameter \( theta_j \), find the most frequently 
        occurring value \( theta_j^* \) across the best runs:

        \[
        theta_j^* = arg\max_{theta_j} \sum_{i=1}^{k} \mathbb{I}(theta_{i,j} = theta_j)
        \]

        This ensures that the selected hyperparameters are robust and frequently associated 
        with low validation loss.

        ## **Why This Approach?**
        
        - It is computationally efficient as it avoids retraining models.
        - It leverages empirical performance data to determine robust hyperparameters.
        - It provides a **stable and reliable** starting point for further tuning (e.g., via Bayesian Optimization).

        ## **Implementation Details**
        
        - Loads past experiment results from CSV.
        - Extracts loss metrics and hyperparameter configurations.
        - Selects the top \( k=10 \) best-performing configurations.
        - Aggregates the most common hyperparameter values from the best runs.
        - Returns a dictionary of optimized hyperparameters.
        """

        file_path = "run_data/wandb_get_runs.csv"
        k = K

        try:
            # Load the CSV file
            df = pd.read_csv(file_path)

            # Convert summary column from string to dictionary
            df['summary'] = df['summary'].apply(ast.literal_eval)

            # Extract relevant loss metrics
            df['val_loss'] = df['summary'].apply(lambda x: x.get('val_loss', float('inf')))
            df['test_loss'] = df['summary'].apply(lambda x: x.get('test_loss', float('inf')))
            df['train_loss'] = df['summary'].apply(lambda x: x.get('train_loss', float('inf')))

            # Extract hyperparameters
            df['hyperparameters'] = df['config'].apply(ast.literal_eval)

            # Select the top 10 best runs based on val_loss
            best_runs = df.nsmallest(k, 'val_loss')

            # Aggregate the best hyperparameters
            hyperparam_counts = {}
            for _, row in best_runs.iterrows():
                for param, value in row['hyperparameters'].items():
                    if param not in hyperparam_counts:
                        hyperparam_counts[param] = []
                    hyperparam_counts[param].append(value)

            # Select the most common values for each hyperparameter
            optimized_hyperparams = {param: max(set(values), key=values.count) for param, values in hyperparam_counts.items()}

            print("\n_opt_best | Optimized Hyperparameters:\n", optimized_hyperparams)

            return optimized_hyperparams
        
        except Exception as e:
            print(f"\n\n_opt_best | Error while optimizing hyperparameters: {e}")
            return {}


# ---------------- OPTIMAL Problem | basyes search ----------------------------------------------- #

    def _opt_bayes(self):
        """
        Optimizes hyperparameters using Bayesian Optimization.

        ## **Mathematical Formulation of the Optimization Problem**

        We aim to solve the following minimization problem:

        \[
        theta* = arg\min_{theta \in Theta} \mathbb{E}[L_text{val}}(theta)]
        \]

        where:

        - \( theta \) represents the set of hyperparameters to be optimized.
        - \( L_text{val}}(theta) \) is the validation loss function (our objective function).
        - \( Theta \) is the feasible set of hyperparameters defined by `sweep_config`.

        ## **Why Bayesian Optimization?**
        
        Since evaluating all possible hyperparameter combinations via **grid search** is impractical, we leverage:
        
        1. **Random Search**: To explore a broad range of parameters.
        2. **Bayesian Optimization**: To refine the search by intelligently selecting promising hyperparameters.

        Bayesian Optimization is advantageous because it models \( L_{text{val}}(theta) \) using a probabilistic **surrogate model**, typically a **Gaussian Process (GP)**:

        \[
        p(L_{text{val}} \mid theta) \sim \mathcal{N}(\mu(theta), \sigma^2(theta))
        \]

        where:
        
        - \( \mu(theta) \) is the mean function, which represents the estimated loss for a given hyperparameter set.
        - \( \sigma^2(theta) \) is the variance function, which represents the uncertainty in the prediction.

        The optimization process involves selecting the next set of hyperparameters \( theta \) by **balancing exploration and exploitation**:

        - **Exploration**: Trying new, unexplored parameter values where uncertainty \( \sigma^2(theta) \) is high.
        - **Exploitation**: Focusing on promising regions where the estimated loss \( \mu(theta) \) is low.

        ## **Acquisition Function: Selecting the Next Parameters**
        
        To decide the next hyperparameter configuration, Bayesian Optimization uses an **acquisition function** \( a(theta) \) that scores potential candidates. One common choice is **Expected Improvement (EI)**:

        \[
        a_{text{EI}}(theta) = \mathbb{E} \left[ \max(0, L_{text{best}} - L_{text{val}}(theta)) right]
        \]

        where \( L_{text{best}} \) is the best loss observed so far.

        ## **Implementation with Scikit-Optimize**
        
        We use `gp_minimize` from **Scikit-Optimize (`skopt`)** to iteratively refine hyperparameter selection based on past observations. The algorithm:
        
        1. **Constructs a probabilistic model** \( p(L_{text{val}} \mid theta) \).
        2. **Uses an acquisition function** to propose new hyperparameters.
        3. **Evaluates the objective function** \( L_{text{val}}(theta) \) for these parameters.
        4. **Updates the model** and repeats the process.

        The final result is a hyperparameter set \( theta^* \) that minimizes the validation loss.

        ---
        
        This function dynamically extracts valid hyperparameter values from past runs and previous best sets. 
        It ensures compatibility with `gp_minimize`, preventing missing or invalid hyperparameter values.
        """


        file_path = "run_data/wandb_get_runs.csv"

        try:
            # Load past results
            df = pd.read_csv(file_path)

            # Convert summary column from string to dictionary
            df['summary'] = df['summary'].apply(ast.literal_eval)

            # Extract loss values
            df['val_loss'] = df['summary'].apply(lambda x: x.get('val_loss', float('inf')))

            # Extract hyperparameters safely
            df['hyperparameters'] = df['config'].apply(ast.literal_eval)

            # Remove rows where 'hyperparameters' is missing
            df = df[df['hyperparameters'].notna()]

            # Select the top N best runs based on val_loss
            top_n = K
            best_runs = df.nsmallest(top_n, 'val_loss')

            # Extract valid hyperparameter values dynamically from best runs
            param_values = {}
            for _, row in best_runs.iterrows():
                for key, value in row['hyperparameters'].items():
                    if key not in param_values:
                        param_values[key] = set()
                    param_values[key].add(value)

            # Ensure predefined sets are included
            prior_sets = PRIOR_SETS

            for set_name in prior_sets:
                if set_name in self.hyperparams_sets:
                    for key, value in self.hyperparams_sets[set_name].items():
                        if key not in param_values:
                            param_values[key] = set()
                        param_values[key].add(value)

            # Ensure 'x_update_mode' exists
            if "x_update_mode" not in param_values:
                param_values["x_update_mode"] = {"alpha", "beta"}  # Default values if missing

            # Fill missing keys with default values
            all_keys = set(param_values.keys())
            for key in all_keys:
                if key not in param_values:
                    param_values[key] = set()
                    param_values[key].add("default_value")  # Change this based on expected type

            # Remove invalid keys (must be string or None)
            param_values = {k: v for k, v in param_values.items() if isinstance(k, str)}

            # Define the hyperparameter search space dynamically
            space = []
            for key, values in param_values.items():
                if isinstance(next(iter(values)), (int, float)):  # Numeric values
                    min_val, max_val = min(values), max(values)
                    if min_val == max_val:  # If only one unique value, force it as Categorical
                        space.append(Categorical([min_val], name=key))
                    else:
                        space.append(Real(min_val, max_val, prior="log-uniform" if min_val > 0 else "uniform", name=key))
                else:  # Categorical values
                    space.append(Categorical(list(values), name=key))

            # Ensure 'x_update_mode' is explicitly present in every row
            df["hyperparameters"] = df["hyperparameters"].apply(lambda x: {**x, "x_update_mode": x.get("x_update_mode", "alpha")})

            # Initialize X_init and Y_init
            X_init = []
            Y_init = []

            # Fill X_init and Y_init with values from best runs
            for _, row in best_runs.iterrows():
                params = row['hyperparameters']
                X_init.append([params.get(key, next(iter(param_values[key]))) for key in param_values])
                Y_init.append(row["val_loss"])

            # Ensure all X_init rows contain all required dimensions
            for i, x in enumerate(X_init):
                while len(x) < len(space):  # Ensure it has all parameters
                    default_value = next(iter(space[len(x)].categories if isinstance(space[len(x)], Categorical) else [space[len(x)].low]))
                    x.append(default_value)

            # Ensure Y_init does not contain 'inf' values
            if len(Y_init) > 0:
                # Remove infinities
                Y_init = [y if np.isfinite(y) else 1.0 for y in Y_init]  
                
                # Cap large values (adjust threshold if needed)
                max_safe_value = 1e3  # Adjust this value as necessary
                Y_init = [min(y, max_safe_value) for y in Y_init]

                # If all values were invalid, set a reasonable default
                if all(y == 1.0 for y in Y_init):  
                    Y_init = [0.1] * len(Y_init)  # Small default loss values

            # Ensure Y_init is not empty (otherwise, skopt will fail)
            if not Y_init:
                Y_init = [0.1]  # Fallback value

                                        
            # Convert Y_init to float64 and ensure all values are finite
            Y_init = np.array(Y_init, dtype=np.float64)

            
            """# Check for invalid values in Y_init
            print(f"NaN in Y_init: {np.isnan(Y_init).sum()}")       # output: 0
            print(f"Inf in Y_init: {np.isinf(Y_init).sum()}")       # output: 0
            print(f"Max value in Y_init: {np.max(Y_init)}")         # output: 0.055643573344226864
            print(f"Min value in Y_init: {np.min(Y_init)}")         # output: 0.021571947232125296


            print("\nX_init before optimization:", X_init)
            print("\nY_init before optimization:", Y_init)"""


            @use_named_args(space)
            def objective(**params):
                # Convert expected integer values to int
                int_keys = ["n_epochs", "batch_size", "es_patience", "decoder_size", "encoder_size", 
                            "decoder_depth", "encoder_depth", "control_rnn_size", "control_rnn_depth"]
                
                for key in int_keys:
                    if key in params:
                        params[key] = int(round(params[key]))

                # Ensure 'x_update_mode' exists
                params.setdefault("x_update_mode", "alpha")

                # Find the closest match instead of exact match
                closest_match = df.iloc[(df["hyperparameters"].apply(lambda x: sum([x.get(k) == v for k, v in params.items()]))).idxmax()]

                if closest_match is not None:
                    return closest_match["val_loss"]
                
                return np.median(Y_init)  # Return median instead of arbitrary 10.0


            # Perform Bayesian Optimization
            res = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=50,  # Increase for better convergence
                n_random_starts=10,  # More exploration before exploitation
                x0=X_init,
                y0=Y_init,
                acq_func="PI",  # Probability of Improvement (more stable than EI)
                random_state=42,
                n_jobs=-1,
                n_restarts_optimizer=10  # Helps with local minima
            )

            
            best_params = dict(zip([dim.name for dim in space], res.x))
            print("\n_opt_bayes | Optimized Hyperparameters:\n", best_params)

            return best_params

        except Exception as e:
            print(f"\n\n_opt_bayes | Error while optimizing hyperparameters: {e}")
            return {}


# --------------------------------------------------------------- #


hp = Hyperparams()
hp._opt_best()           # using the top 10 best runs
hp._opt_bayes()          # using bayesian optimization