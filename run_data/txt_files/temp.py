"""
---------------------------------------
Order by min:
---------------------------------------


Top 30 results by best_val:
                                      val_loss  best_val  best_test  best_train  epoch
vdp_test0___005--BE_B-126-entropy-22  0.034220  0.030506   0.055751    0.014901     68 29
                          vdp_TEST-6  0.039730  0.031103   0.031672    0.010110     54 13
                          vdp_data11  0.032431  0.032027   0.071255    0.007585     87 24
                        vdp_TEST-r_1  0.043439  0.034618   0.027133    0.009729     84 10
                          vdp_TEST-3  0.041426  0.036084   0.062296    0.006991     66 15
            vdp_005--exact_B-entropy  0.039655  0.037643   0.035559    0.016377     73 30
                               VDP-4  0.050824  0.037681   0.026898    0.009864     76 25
                        TEMP-1-data3  0.042830  0.038221   0.043928    0.006452    105  6
                     vdp_test3-data3  0.044685  0.038330   0.023360    0.010910     44 27
                        vdp_TEST-r_3  0.050480  0.038446   0.066855    0.014678     56 12
                         vdp__lpv-TU  0.052758  0.039347   0.048107    0.023143     37 18
                      TEMP-without-2  0.039610  0.039426   0.031179    0.013388     42  7

                              best-5  0.047489  0.040723   0.191966    0.007684     66  1
                     vdp_test3-data1  0.060579  0.040782   0.080850    0.018978     43 28
                          vdp_TEST-1  0.048545  0.041286   0.065949    0.011451     61 17
                          vdp_TEST-2  0.046214  0.041582   0.127981    0.017602     43 16
                         vdp__lpv-BE  0.048493  0.041879   0.024641    0.006103     83 19
                        vdp_TEST-r_2  0.049006  0.043399   0.037774    0.012601     55 11
                     vdp__new-dis_FE  0.048795  0.043460   0.030123    0.008976    103 20
                               VDP-3  0.043710  0.043710   0.026717    0.011442     34 26
                          vdp_TEST-4  0.068428  0.043995   0.077090    0.020715     43 14
                         TEMP-with-2  0.052379  0.044202   0.032497    0.012957     42  8
                           TEMP-with  0.060862  0.044223   0.046377    0.019144     28  9
                                 RK4  0.047118  0.044900   0.083777    0.008219     50  4
                     vdp__new-dis_TU  0.058871  0.045001   0.054385    0.028403     39 21
                  vdp__new-dis_exact  0.057902  0.045180   0.036718    0.010145     72 22
                              best-3  0.051896  0.045438   0.151205    0.008742     59  2
                     vdp__new-dis_BE  0.050505  0.046924   0.029218    0.005297     99 23
                              best-1  0.051546  0.047589   0.120988    0.006839     67  3
                        TEMP-2-data3  0.051975  0.048158   0.033036    0.027025     50  5
"""


import pandas as pd
import os


class check: 
    def __init__(self):

        metrics_config = [
            #"discretisation_mode",  # TU 
            #"encoder_size",         # 2
            #"sched_patience",       # 10
            #"control_rnn_size",     # 48
            #"linearisation_mode",   # lpv
            #"batch_size",           # 96
            #'lr',                   # 0.001
            #'es_delta',             # 1e-07
            #'es_patience',          # 25
            #'decoder_size',         # 2
            #'sched_factor',         # 3
            #'decoder_depth',        # 2
            #'encoder_depth',        # 2
            #'x_update_mode',        # entropy
            ]
        
        metrics_summary = [
            "best_val",
            "best_test",
            "best_train",
            ]

        self.list_of_function = [
            "vdp_test0___005__BE_B_126_entropy_22", 
            "vdp_TEST_6",
            "vdp_data11",
            "vdp_TEST_r_1",
            "vdp_TEST_3",
            "vdp_005__exact_B_entropy",
            "VDP_4",
            "TEMP_1_data3",
            "vdp_test3_data3",
            "vdp_TEST_r_3",
            "vdp__lpv_TU",
            "TEMP_without_2",
            ]
        
        for metric in metrics_config:   self.get(metric, "config")
        #for metric in metrics_summary:  self.get(metric, "summary")
        
    
    def get(self, metric, mode="config"):
        print(f"\n\n--- Metric: {metric} ---\n")
        for function in self.list_of_function:
            # Get the function from the class
            func = getattr(self, function)
            # Call the function to get Summary, Config, and name
            Summary, Config, name = func()
            
            # Convert Summary and Config to DataFrames
            df_summary = pd.DataFrame([Summary])
            df_config = pd.DataFrame([Config])
            
            # Concatenate DataFrames
            df_combined = pd.concat([df_summary, df_config], axis=1)
            
            if mode == "config":    print(f"{name:<40} : {Config[metric]}")
            else:                   print(f"{name:<40} : {Summary[metric]}")


# -------------------------------------------------------------
    def vdp_test0___005__BE_B_126_entropy_22(self): 
        Summary = {
            '_runtime': 34629.077886309, 
            '_step': 68, 
            '_timestamp': 1743115702.5598369, 
            '_wandb': {'runtime': 34630}, 
            'batch_size': 126, 
            'best_epoch': 48, 
            'best_test': 0.05575081062852405, 
            'best_train': 0.014900697957879553, 
            'best_val': 0.030505849921610206, 
            'epoch': 68, 
            'lr': 0.00025, 
            'n_epochs': 200, 
            'test_loss': 0.04379362871986814, 
            'time': 34497.40789461136, 
            'train_loss': 0.011582090097363109, 
            'val_loss': 0.03422026717453264
            }
        
        Config = {
            'lr': 0.001, 
            'loss': 'mse', 
            'es_delta': 1e-07, 
            'mode_dnn': 'FFNet', 
            'mode_rnn': 'new', 
            'n_epochs': 200, 
            'batch_size': 126, 
            'es_patience': 20, 
            'decoder_size': 2, 
            'encoder_size': 2, 
            'sched_factor': 2, 
            'decoder_depth': 2, 
            'encoder_depth': 2, 
            'x_update_mode': 'entropy', 
            'optimiser_mode': 'adam', 
            'sched_patience': 10, 
            'control_rnn_size': 22, 
            'control_rnn_depth': 1, 
            'discretisation_mode': 'BE',
            'linearisation_mode': 'static'
            }
        
        name = "vdp_test0___005--BE_B-126-entropy-22"

        return Summary, Config, name

    def vdp_TEST_6(self):
        Summary = {
            '_runtime': 44317.0104858,
            '_step': 54,
            '_timestamp': 1743692122.4983504,
            '_wandb': {'runtime': 44465},
            'batch_size': 96,
            'best_epoch': 38,
            'best_test': 0.03167172134410413,
            'best_train': 0.010110082938188008,
            'best_val': 0.03110319181966285,
            'epoch': 54,
            'lr': 0.0001111111111111111,
            'n_epochs': 200,
            'test_loss': 0.034154415574102176,
            'time': 44189.57351207733,
            'train_loss': 0.007498276012483984,
            'val_loss': 0.03972955086889366
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 96,
            'es_patience': 25,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 3,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 48,
            'control_rnn_depth': 1,
            'linearisation_mode': 'lpv',
            'discretisation_mode': 'TU', 
        }
        name = "vdp_TEST-6"
        return Summary, Config, name

    def vdp_data11(self):
        Summary = {
            '_runtime': 33384.4049752,
            '_step': 87,
            '_timestamp': 1743342657.4427962,
            '_wandb': {'runtime': 33519},
            'batch_size': 96,
            'best_epoch': 67,
            'best_test': 0.07125506124326161,
            'best_train': 0.007585044405127447,
            'best_val': 0.03202745840618653,
            'epoch': 87,
            'lr': 3.703703703703703e-05,
            'n_epochs': 200,
            'test_loss': 0.07165112284322579,
            'time': 33213.523649930954,
            'train_loss': 0.006260095416412999,
            'val_loss': 0.03243069303044606
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 96,
            'es_patience': 25,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 3,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 48,
            'control_rnn_depth': 1,
            'discretisation_mode': 'BE', 
            'linearisation_mode': 'static'
        }
        name = "vdp_data11"
        return Summary, Config, name
    
    def vdp_TEST_r_1(self):
        Summary = {
            '_runtime': 53725.238367715,
            '_step': 84,
            '_timestamp': 1743746230.0750244,
            '_wandb': {'runtime': 53726},
            'batch_size': 96,
            'best_epoch': 59,
            'best_test': 0.027132561363812004,
            'best_train': 0.009729017913785007,
            'best_val': 0.034618378561433585,
            'epoch': 84,
            'lr': 1.2345679012345677e-05,
            'n_epochs': 200,
            'test_loss': 0.02573979173653892,
            'time': 53542.17037367821,
            'train_loss': 0.005448304720976878,
            'val_loss': 0.04343893308015097
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 96,
            'es_patience': 25,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 3,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 48,
            'control_rnn_depth': 1,
            'linearisation_mode': 'lpv',
            'discretisation_mode': 'TU', 
        }
        name = "vdp_TEST-r_1"
        return Summary, Config, name

    def vdp_TEST_3(self):
        Summary = {
            '_runtime': 67359.402882456,
            '_step': 66,
            '_timestamp': 1743714288.3201623,
            '_wandb': {'runtime': 68362},
            'batch_size': 96,
            'best_epoch': 59,
            'best_test': 0.0622958194774886,
            'best_train': 0.0069914077083388015,
            'best_val': 0.03608399536460638,
            'epoch': 66,
            'lr': 0.0001111111111111111,
            'n_epochs': 200,
            'test_loss': 0.06578786258718797,
            'time': 67191.70189142227,
            'train_loss': 0.007358250214666542,
            'val_loss': 0.04142646810838154
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 96,
            'es_patience': 25,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 3,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 48,
            'control_rnn_depth': 1,
            'linearisation_mode': 'lpv',
            'discretisation_mode': 'TU', 
        }
        name = "vdp_TEST-3"
        return Summary, Config, name

    def vdp_005__exact_B_entropy(self):
        Summary = {
            '_runtime': 30868.802875179,
            '_step': 73,
            '_timestamp': 1743103984.6789374,
            '_wandb': {'runtime': 30870},
            'batch_size': 256,
            'best_epoch': 53,
            'best_test': 0.035558891657274216,
            'best_train': 0.01637718313814778,
            'best_val': 0.037642860086634755,
            'epoch': 73,
            'lr': 0.00025,
            'n_epochs': 200,
            'test_loss': 0.034320139326155186,
            'train_loss': 0.011976932028406544,
            'val_loss': 0.03965486912056804
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 256,
            'es_patience': 20,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 2,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 22,
            'control_rnn_depth': 1,
            'discretisation_mode': 'exact', 
            'linearisation_mode': 'static'
        }
        name = "vdp_005--exact_B-entropy"
        return Summary, Config, name

    def VDP_4(self):
        Summary = {
            '_runtime': 23989.6479779,
            '_step': 76,
            '_timestamp': 1743276188.5109458,
            '_wandb': {'runtime': 23997},
            'batch_size': 96,
            'best_epoch': 54,
            'best_test': 0.02689770906276646,
            'best_train': 0.00986370194466814,
            'best_val': 0.03768074515807841,
            'epoch': 76,
            'lr': 3.703703703703703e-05,
            'n_epochs': 200,
            'test_loss': 0.024507656205622924,
            'train_loss': 0.006626730716402923,
            'val_loss': 0.05082417292786496
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 96,
            'es_patience': 25,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 3,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 48,
            'control_rnn_depth': 1,
            'discretisation_mode': 'BE', 
            'linearisation_mode': 'static'
        }
        name = "VDP-4"
        return Summary, Config, name

    def TEMP_1_data3(self):
        Summary = {
            '_runtime': 53722.3176586,
            '_step': 105,
            '_timestamp': 1744264915.0027986,
            '_wandb': {'runtime': 53724},
            'batch_size': 128,
            'best_epoch': 80,
            'best_test': 0.04392782938740556,
            'best_train': 0.0064521499061907725,
            'best_val': 0.0382206335013348,
            'epoch': 105,
            'lr': 1.2345679012345677e-05,
            'n_epochs': 200,
            'test_loss': 0.04485671254732306,
            'train_loss': 0.005439786472057224,
            'val_loss': 0.042829702741333416
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 128,
            'es_patience': 25,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 3,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 48,
            'control_rnn_depth': 1,
            'linearisation_mode': 'lpv',
            'discretisation_mode': 'TU', 
        }
        name = "TEMP-1-data3"
        return Summary, Config, name

    def vdp_test3_data3(self):
        Summary = {
            '_runtime': 29007.1377397,
            '_step': 44,
            '_timestamp': 1743198713.494727,
            '_wandb': {'runtime': 29007},
            'batch_size': 96,
            'best_epoch': 43,
            'best_test': 0.023360299711514796,
            'best_train': 0.010909817778387124,
            'best_val': 0.03832991088607481,
            'epoch': 44,
            'lr': 0.0003333333333333333,
            'n_epochs': 200,
            'test_loss': 0.031105763190204187,
            'train_loss': 0.011951466800556296,
            'val_loss': 0.04468533784771959
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 96,
            'es_patience': 25,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 3,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 48,
            'control_rnn_depth': 1,
            'discretisation_mode': 'BE', 
            'linearisation_mode': 'static'
        }
        name = "vdp_test3-data3"
        return Summary, Config, name

    def vdp_TEST_r_3(self):
        Summary = {
            '_runtime': 37317.6401208,
            '_step': 56,
            '_timestamp': 1743718670.4868498,
            '_wandb': {'runtime': 37319},
            'batch_size': 96,
            'best_epoch': 31,
            'best_test': 0.06685450981326756,
            'best_train': 0.014677851266848544,
            'best_val': 0.03844641860840576,
            'epoch': 56,
            'lr': 0.0001111111111111111,
            'n_epochs': 200,
            'test_loss': 0.04228987147854198,
            'train_loss': 0.0070646602100515295,
            'val_loss': 0.05047990973772747
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 96,
            'es_patience': 25,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 3,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 48,
            'control_rnn_depth': 1,
            'linearisation_mode': 'lpv',
            'discretisation_mode': 'TU', 
        }
        name = "vdp_TEST-r_3"
        return Summary, Config, name

    def vdp__lpv_TU(self):
        Summary = {
            '_runtime': 24210.5472237,
            '_step': 37,
            '_timestamp': 1743529255.7937255,
            '_wandb': {'runtime': 24212},
            'batch_size': 96,
            'best_epoch': 12,
            'best_test': 0.04810731543139333,
            'best_train': 0.023143090658806383,
            'best_val': 0.03934687344978253,
            'epoch': 37,
            'lr': 0.0001111111111111111,
            'n_epochs': 200,
            'test_loss': 0.03602759371555987,
            'train_loss': 0.008779073801142947,
            'val_loss': 0.05275846283794159
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 96,
            'es_patience': 25,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 3,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 48,
            'control_rnn_depth': 1,
            'discretisation_mode': 'TU', 
            'linearisation_mode': 'lpv'
        }
        name = "vdp__lpv-TU"
        return Summary, Config, name

    def TEMP_without_2(self):
        Summary = {
            '_runtime': 25211.7094229,
            '_step': 42,
            '_timestamp': 1744164887.6110842,
            '_wandb': {'runtime': 25211},
            'batch_size': 128,
            'best_epoch': 41,
            'best_test': 0.03117914146019353,
            'best_train': 0.013388419893407631,
            'best_val': 0.039426057496004634,
            'epoch': 42,
            'lr': 0.0003333333333333333,
            'n_epochs': 200,
            'test_loss': 0.029815875082498507,
            'train_loss': 0.012941041680437231,
            'val_loss': 0.0396100790609443
        }
        Config = {
            'lr': 0.001,
            'loss': 'mse',
            'es_delta': 1e-07,
            'mode_dnn': 'FFNet',
            'mode_rnn': 'new',
            'n_epochs': 200,
            'batch_size': 128,
            'es_patience': 25,
            'decoder_size': 2,
            'encoder_size': 2,
            'sched_factor': 3,
            'decoder_depth': 2,
            'encoder_depth': 2,
            'x_update_mode': 'entropy',
            'optimiser_mode': 'adam',
            'sched_patience': 10,
            'control_rnn_size': 48,
            'control_rnn_depth': 1,
            'linearisation_mode': 'lpv',
            'discretisation_mode': 'TU'
        }
        name = "TEMP-without-2"
        return Summary, Config, name
# --------------------------------------------------------------


if __name__ == "__main__":
    check()