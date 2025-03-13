import numpy as np


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
            'hyperparams___swift_sweep_1': {'lr': 0.0005, 'loss': 'mse', 'es_delta': 1e-07, 'n_epochs': 500, 'batch_size': 256, 'es_patience': 20, 'decoder_size': 1, 'encoder_size': 2, 'sched_factor': 2, 'decoder_depth': 1, 'encoder_depth': 1, 'x_update_mode': 'beta', 'optimiser_mode': 'adam', 'sched_patience': 10, 'control_rnn_size': 12, 'control_rnn_depth': 1, 'discretisation_mode': 'TU'}  
        }
    
    
    def get_hyperparams(self, name):
        return self.hyperparams_sets.get(name, f"Hyperparameter set '{name}' not found.")

# Example usage:
hp = Hyperparams()
print(hp.get_hyperparams('hyperparams___set_1'))