import pandas as pd
import matplotlib.pyplot as plt



class CalcValues:
    def __init__(self):
        self.datasets = self.DataSet()

        self.metrics = [
            "_step", 
            "best_epoch", 
            "best_test", 
            "best_train", 
            "best_val", 
            "lr", 
            "test_loss", 
            "time", 
            "train_loss", 
            "val_loss"
            ]

        #for name, data in self.datasets.items():    self.display_results(name, data)
        #self.display_final_comparison__best()
        #self.display_final_comparison__mean()

        for param in ["test_loss", "train_loss", "val_loss"]: self.plot_metric_across_datasets(self.datasets, param)



    def DataSet(self):
        default_code_same_dim = [
            {'run': "017", '_runtime': 7546.45702, '_step': 117, '_timestamp': 1741182593.7156532, '_wandb': {'runtime': 7546}, 'best_epoch': 98, 'best_test': 0.03409814938075013, 'best_train': 0.018149984931504284, 'best_val': 0.028380416333675385, 'epoch': 118, 'lr': 0.000125, 'test_loss': 0.03472252781428988, 'time': 7528.629385232925, 'train_loss': 0.015891212812334143, 'val_loss': 0.03212327557423758}
            , {'run': "026", '_runtime': 7346.7665658, '_step': 106, '_timestamp': 1741430459.320042, '_wandb': {'runtime': 7346}, 'best_epoch': 87, 'best_test': 0.025843684265892657, 'best_train': 0.014283652101993245, 'best_val': 0.02398856176388642, 'epoch': 107, 'lr': 0.000125, 'test_loss': 0.026321270325708957, 'time': 7328.914924144745, 'train_loss': 0.014671892942811446, 'val_loss': 0.026048427477242456}
            , {'run': "027", '_runtime': 8290.5964382, '_step': 126, '_timestamp': 1741431693.8628545, '_wandb': {'runtime': 8290}, 'best_epoch': 107, 'best_test': 0.027500419862686643, 'best_train': 0.01666778990772193, 'best_val': 0.05470035585855681, 'epoch': 127, 'lr': 0.00025, 'test_loss': 0.027228796898963903, 'time': 8272.850373268127, 'train_loss': 0.014251226118258223, 'val_loss': 0.06482056906772038}
        ]

        new_LSTM = [
            {'run': "005", '_runtime': 6233.088474, '_step': 31, '_timestamp': 1740153810.4094262, '_wandb': {'runtime': 6341}, 'best_epoch': 31, 'best_test': 0.046496183951459234, 'best_train': 0.02178202904070969, 'best_val': 0.042363496674668225, 'epoch': 32, 'lr': 0.001, 'test_loss': 0.04526456543022678, 'time': 6205.528746366501, 'train_loss': 0.02056183809877703, 'val_loss': 0.04706020126976664}
            , {'run': "006", '_runtime': 6107.010322916, '_step': 64, '_timestamp': 1740337413.0724814, '_wandb': {'runtime': 6107}, 'best_epoch': 45, 'best_test': 0.06412715399785647, 'best_train': 0.016991450096524897, 'best_val': 0.05093690377497484, 'epoch': 65, 'lr': 0.0005, 'test_loss': 0.05433392481848834, 'time': 6086.771798610687, 'train_loss': 0.0092004129968623, 'val_loss': 0.05783890521833821}
            , {'run': "007(1)", '_runtime': 9218.6605851, '_step': 60, '_timestamp': 1740426466.3221452, '_wandb': {'runtime': 9218}, 'best_epoch': 61, 'best_test': 0.11439969715854478, 'best_train': 0.02425803190402726, 'best_val': 0.04724688281024259, 'epoch': 61, 'lr': 0.001, 'test_loss': 0.11439969715854478, 'time': 9199.148099899292, 'train_loss': 0.02425803190402726, 'val_loss': 0.04724688281024259}
            , {'run': "007(2)", '_runtime': 9218.6605851, '_step': 60, '_timestamp': 1740426466.3221452, '_wandb': {'runtime': 9218}, 'best_epoch': 61, 'best_test': 0.11439969715854478, 'best_train': 0.02425803190402726, 'best_val': 0.04724688281024259, 'epoch': 61, 'lr': 0.001, 'test_loss': 0.11439969715854478, 'time': 9199.148099899292, 'train_loss': 0.02425803190402726, 'val_loss': 0.04724688281024259}
            , {'run': "007(3)", '_runtime': 12590.5137433, '_step': 66, '_timestamp': 1740429838.1752734, '_wandb': {'runtime': 12590}, 'best_epoch': 63, 'best_test': 0.08391683063809834, 'best_train': 0.023757762576221787, 'best_val': 0.03939290060883477, 'epoch': 67, 'lr': 0.001, 'test_loss': 0.0924254282717667, 'time': 12571.0012280941, 'train_loss': 0.02796114632790839, 'val_loss': 0.045097659888958176}
            , {'run': "007(4)", '_runtime': 12590.5137433, '_step': 66, '_timestamp': 1740429838.1752734, '_wandb': {'runtime': 12892}, 'best_epoch': 63, 'best_test': 0.08391683063809834, 'best_train': 0.023757762576221787, 'best_val': 0.03939290060883477, 'epoch': 67, 'lr': 0.001, 'test_loss': 0.0924254282717667, 'time': 12571.0012280941, 'train_loss': 0.02796114632790839, 'val_loss': 0.045097659888958176}
            , {'run': "007(5)", '_runtime': 12892.354965, '_step': 67, '_timestamp': 1740430140.015451, '_wandb': {'runtime': 12892}, 'best_epoch': 63, 'best_test': 0.08391683063809834, 'best_train': 0.023757762576221787, 'best_val': 0.03939290060883477, 'epoch': 68, 'lr': 0.001, 'test_loss': 0.11099068507079096, 'time': 12872.841405630112, 'train_loss': 0.02273536986971974, 'val_loss': 0.04870927724100295}
            , {'run': "007(6)", '_runtime': 12892.354965, '_step': 67, '_timestamp': 1740430140.015451, '_wandb': {'runtime': 12892}, 'best_epoch': 63, 'best_test': 0.08391683063809834, 'best_train': 0.023757762576221787, 'best_val': 0.03939290060883477, 'epoch': 68, 'lr': 0.001, 'test_loss': 0.11099068507079096, 'time': 12872.841405630112, 'train_loss': 0.02273536986971974, 'val_loss': 0.04870927724100295}
            , {'run': "007(7)", '_runtime': 13228.3679994, '_step': 68, '_timestamp': 1740430476.0301545, '_wandb': {'runtime': 13228}, 'best_epoch': 63, 'best_test': 0.08391683063809834, 'best_train': 0.023757762576221787, 'best_val': 0.03939290060883477, 'epoch': 69, 'lr': 0.001, 'test_loss': 0.08817817117013628, 'time': 13208.855578899384, 'train_loss': 0.025686304326410645, 'val_loss': 0.049629264141595554}
            , {'run': "007(8)", '_runtime': 13228.3679994, '_step': 68, '_timestamp': 1740430476.0301545, '_wandb': {'runtime': 13228}, 'best_epoch': 63, 'best_test': 0.08391683063809834, 'best_train': 0.023757762576221787, 'best_val': 0.03939290060883477, 'epoch': 69, 'lr': 0.001, 'test_loss': 0.08817817117013628, 'time': 13208.855578899384, 'train_loss': 0.025686304326410645, 'val_loss': 0.049629264141595554}
            , {'run': "007(9)", '_runtime': 13228.3679994, '_step': 68, '_timestamp': 1740430476.0301545, '_wandb': {'runtime': 13228}, 'best_epoch': 63, 'best_test': 0.08391683063809834, 'best_train': 0.023757762576221787, 'best_val': 0.03939290060883477, 'epoch': 69, 'lr': 0.001, 'test_loss': 0.08817817117013628, 'time': 13208.855578899384, 'train_loss': 0.025686304326410645, 'val_loss': 0.049629264141595554}
            , {'run': "009(5)", '_runtime': 6920.9206076, '_step': 43, '_timestamp': 1740507916.2359128, '_wandb': {'runtime': 6920}, 'best_epoch': 43, 'best_test': 0.05673211369486082, 'best_train': 0.03496907723367845, 'best_val': 0.2796749094175914, 'epoch': 44, 'lr': 0.001, 'test_loss': 0.058428846299648285, 'time': 6897.475360393524, 'train_loss': 0.029711594164529176, 'val_loss': 0.28459808395968544}
            , {'run': "009(2)", '_runtime': 7290.709749, '_step': 45, '_timestamp': 1740508286.025054, '_wandb': {'runtime': 7290}, 'best_epoch': 45, 'best_test': 0.057728635649832466, 'best_train': 0.02921068677727981, 'best_val': 0.27701510891081793, 'epoch': 46, 'lr': 0.001, 'test_loss': 0.05660884188754218, 'time': 7267.264501571655, 'train_loss': 0.05261738786582278, 'val_loss': 0.2823867861713682}
            , {'run': "009(3)", '_runtime': 8055.6265541, '_step': 50, '_timestamp': 1740509050.941317, '_wandb': {'runtime': 8055}, 'best_epoch': 50, 'best_test': 0.052941458270190256, 'best_train': 0.026101484303436583, 'best_val': 0.26213606121757677, 'epoch': 51, 'lr': 0.001, 'test_loss': 0.05675290956620186, 'time': 8032.18076467514, 'train_loss': 0.026066893007034665, 'val_loss': 0.26720066214837723}
            , {'run': "009(4)", '_runtime': 17349.091133, '_step': 98, '_timestamp': 1740518344.4053724, '_wandb': {'runtime': 17493}, 'best_epoch': 97, 'best_test': 0.04557893265570913, 'best_train': 0.017039695917259134, 'best_val': 0.23109009235151232, 'epoch': 99, 'lr': 0.0005, 'test_loss': 0.04779352785812484, 'time': 17325.6448199749, 'train_loss': 0.01854242477566004, 'val_loss': 0.2401606762219989}
        ]

        x_update_alpha = [
            {'run': "015", '_runtime': 11637.674524, '_step': 65, '_timestamp': 1741122288.162348, '_wandb': {'runtime': 11637}, 'best_epoch': 56, 'best_test': 0.15566413057228876, 'best_train': 0.03127661450869507, 'best_val': 0.0726061523670242, 'epoch': 66, 'lr': 0.001, 'test_loss': 0.17950638903984947, 'time': 11612.085956335068, 'train_loss': 0.0874257475413658, 'val_loss': 0.16335126840405997}
            , {'run': "016", '_runtime': 26602.2111636, '_step': 108, '_timestamp': 1741157380.6444006, '_wandb': {'runtime': 26602}, 'best_epoch': 89, 'best_test': 0.05378690220060803, 'best_train': 0.020549435986491737, 'best_val': 0.052191075676726914, 'epoch': 109, 'lr': 0.000125, 'test_loss': 0.04970564223116353, 'time': 26571.87525510788, 'train_loss': 0.01790892477171919, 'val_loss': 0.055643573344226864}
            , {'run': "018(1)", '_runtime': 12845.8606273, '_step': 84, '_timestamp': 1741201116.3583372, '_wandb': {'runtime': 12845}, 'best_epoch': 80, 'best_test': 0.09021028982741491, 'best_train': 0.020383733487318433, 'best_val': 0.055468578689864705, 'epoch': 85, 'lr': 0.001, 'test_loss': 0.11940238163584754, 'time': 12816.142216682434, 'train_loss': 0.02383146670563196, 'val_loss': 0.0593999158591032}
            , {'run': "018(2-3)", '_runtime': 29190.6178727, '_step': 156, '_timestamp': 1741217461.1155827, '_wandb': {'runtime': 29190}, 'best_epoch': 137, 'best_test': 0.08476408736573325, 'best_train': 0.01709923068327563, 'best_val': 0.04871034687237134, 'epoch': 157, 'lr': 0.000125, 'test_loss': 0.08193321597008478, 'time': 29160.899462223053, 'train_loss': 0.01672123080838925, 'val_loss': 0.04979510998560323}
            , {'run': "019", '_runtime': 23343.4636111, '_step': 142, '_timestamp': 1741280362.9845507, '_wandb': {'runtime': 23343}, 'best_epoch': 123, 'best_test': 0.0533087533558645, 'best_train': 0.02122072688249684, 'best_val': 0.25057389338811237, 'epoch': 143, 'lr': 6.25e-05, 'test_loss': 0.05645283382563364, 'time': 23315.39107489586, 'train_loss': 0.01923207267034779, 'val_loss': 0.25213538606961566}
            , {'run': "020", '_runtime': 19282.2622103, '_step': 109, '_timestamp': 1741293582.378608, '_wandb': {'runtime': 19282}, 'best_epoch': 90, 'best_test': 0.1449957171839381, 'best_train': 0.02281889265708665, 'best_val': 0.03265471753501703, 'epoch': 110, 'lr': 0.000125, 'test_loss': 0.14242024950328327, 'time': 19247.294664382935, 'train_loss': 0.019359053211119123, 'val_loss': 0.03382213047099492}
            , {'run': "021", '_runtime': 24220.1446849, '_step': 145, '_timestamp': 1741321179.491952, '_wandb': {'runtime': 24220}, 'best_epoch': 126, 'best_test': 0.06649408028239295, 'best_train': 0.018753742099439027, 'best_val': 0.11177096456762343, 'epoch': 146, 'lr': 3.125e-05, 'test_loss': 0.06538293534328067, 'time': 24190.640423059464, 'train_loss': 0.017817277136066602, 'val_loss': 0.1155335587405023}
        ]

        x_update_alpha_opt = [
            {'run': "022", '_runtime': 15588.0104955, '_step': 79, '_timestamp': 1741373321.7143245, '_wandb': {'runtime': 15588}, 'best_epoch': 60, 'best_test': 0.19954035431146624, 'best_train': 0.033677110980663984, 'best_val': 0.1268843865347287, 'epoch': 80, 'lr': 0.0005, 'test_loss': 0.17315763891452834, 'time': 15557.825784683228, 'train_loss': 0.02871229724771289, 'val_loss': 0.13715681126193394}
            , {'run': "023", '_runtime': 19091.3150562, '_step': 90, '_timestamp': 1741383918.291978, '_wandb': {'runtime': 19091}, 'best_epoch': 71, 'best_test': 0.17176606902290906, 'best_train': 0.03029533599813779, 'best_val': 0.0722720061857549, 'epoch': 91, 'lr': 0.00025, 'test_loss': 0.15187193074869731, 'time': 19058.01224398613, 'train_loss': 0.021802850865892003, 'val_loss': 0.08129255662834833}
            , {'run': "024", '_runtime': 16780.6603587, '_step': 74, '_timestamp': 1741391706.1888623, '_wandb': {'runtime': 16780}, 'best_epoch': 55, 'best_test': 0.29205070069384953, 'best_train': 0.030286846978087274, 'best_val': 0.058476886756363367, 'epoch': 75, 'lr': 0.00025, 'test_loss': 0.20884731518370764, 'time': 16748.41609811783, 'train_loss': 0.024770572751997007, 'val_loss': 0.06078948893599094}
            , {'run': "025", '_runtime': 6466.9516759, '_step': 47, '_timestamp': 1741404462.7291248, '_wandb': {'runtime': 6466}, 'best_epoch': 28, 'best_test': 0.2853936068122349, 'best_train': 0.11861003164655318, 'best_val': 0.1437250425418218, 'epoch': 48, 'lr': 0.0005, 'test_loss': 0.3043967697118956, 'time': 6438.592477083206, 'train_loss': 0.05953135157112407, 'val_loss': 0.1697242379425064}
        ]

        x_update_beta = [
            {'run': "028(n.s.)", '_runtime': 8943.1548878, '_step': 56, '_timestamp': 1741480930.8972335, '_wandb': {'runtime': 9085}, 'best_epoch': 57, 'best_test': 0.09263929032853672, 'best_train': 0.03767314413316035, 'best_val': 0.16285020195775562, 'epoch': 57, 'lr': 0.0005, 'test_loss': 0.09263929032853672, 'time': 8914.816588163376, 'train_loss': 0.03767314413316035, 'val_loss': 0.16285020195775562}
            , {'run': "028", '_runtime': 8943.1548878, '_step': 56, '_timestamp': 1741480930.8972335, '_wandb': {'runtime': 9085}, 'best_epoch': 57, 'best_test': 0.09263929032853672, 'best_train': 0.03767314413316035, 'best_val': 0.16285020195775562, 'epoch': 57, 'lr': 0.0005, 'test_loss': 0.09263929032853672, 'time': 8914.816588163376, 'train_loss': 0.03767314413316035, 'val_loss': 0.16285020195775562}
            , {'run': "029(n.s.)", '_runtime': 14979.936391781, '_step': 124, '_timestamp': 1741489422.5467255, '_wandb': {'runtime': 14979}, 'best_epoch': 105, 'best_test': 0.1153069618379786, 'best_train': 0.01781458006531158, 'best_val': 0.0347165379318453, 'epoch': 125, 'lr': 0.000125, 'test_loss': 0.11321590101671596, 'time': 14932.54727458954, 'train_loss': 0.017040841759394403, 'val_loss': 0.03776116819963569}
            , {'run': "029", '_runtime': 14979.936391781, '_step': 124, '_timestamp': 1741489422.5467255, '_wandb': {'runtime': 14979}, 'best_epoch': 105, 'best_test': 0.1153069618379786, 'best_train': 0.01781458006531158, 'best_val': 0.0347165379318453, 'epoch': 125, 'lr': 0.000125, 'test_loss': 0.11321590101671596, 'time': 14932.54727458954, 'train_loss': 0.017040841759394403, 'val_loss': 0.03776116819963569}
            , {'run': "030(n.s.)", '_runtime': 9795.948917055, '_step': 77, '_timestamp': 1741501099.559984, '_wandb': {'runtime': 9795}, 'best_epoch': 58, 'best_test': 0.12283636759670954, 'best_train': 0.025061399976491296, 'best_val': 0.06987644891653742, 'epoch': 78, 'lr': 0.0005, 'test_loss': 0.11164449415509664, 'time': 9755.34075474739, 'train_loss': 0.027246853849165652, 'val_loss': 0.08021583975780577}
            , {'run': "030", '_runtime': 9795.948917055, '_step': 77, '_timestamp': 1741501099.559984, '_wandb': {'runtime': 9795}, 'best_epoch': 58, 'best_test': 0.12283636759670954, 'best_train': 0.025061399976491296, 'best_val': 0.06987644891653742, 'epoch': 78, 'lr': 0.0005, 'test_loss': 0.11164449415509664, 'time': 9755.34075474739, 'train_loss': 0.027246853849165652, 'val_loss': 0.08021583975780577}
            , {'run': "031", '_runtime': 14129.219967, '_step': 85, '_timestamp': 1741515333.7657564, '_wandb': {'runtime': 14129}, 'best_epoch': 66, 'best_test': 0.06133903092926457, 'best_train': 0.0189332693283047, 'best_val': 0.11545243593198912, 'epoch': 86, 'lr': 0.000125, 'test_loss': 0.053553428796548695, 'time': 14094.317236423492, 'train_loss': 0.01907580905155372, 'val_loss': 0.11624801980834158}
            , {'run': "032", '_runtime': 8547.257474084, '_step': 71, '_timestamp': 1741519391.8302178, '_wandb': {'runtime': 8547}, 'best_epoch': 52, 'best_test': 0.11412586821686654, 'best_train': 0.03626639311196943, 'best_val': 0.07681829204398488, 'epoch': 72, 'lr': 0.00025, 'test_loss': 0.16247747891715594, 'time': 8527.486145973206, 'train_loss': 0.0253108320807023, 'val_loss': 0.10421322646831709}
            , {'run': "033", '_runtime': 11596.988310016, '_step': 89, '_timestamp': 1741556368.7455869, '_wandb': {'runtime': 11596}, 'best_epoch': 70, 'best_test': 0.025469546516736347, 'best_train': 0.020823853031273872, 'best_val': 0.05242124152561975, 'epoch': 90, 'lr': 0.00025, 'test_loss': 0.028552606967943057, 'time': 11577.355808258057, 'train_loss': 0.017268735077724886, 'val_loss': 0.05523285769399197}
        ]

        return{
            "default_code_same_dim": default_code_same_dim,
            #"new_LSTM": new_LSTM,
            #"x_update_alpha": x_update_alpha,
            "x_update_alpha_opt": x_update_alpha_opt,
            "x_update_beta": x_update_beta
            }


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

    def calculate_average_metrics(self, data_list):
        """
        Computes and returns a dataframe with the average values for each metric.
        """
        if not data_list:
            return None
        
        df = pd.DataFrame(data_list)

        # Select only numerical columns
        numeric_columns = self.metrics
        
        # Compute mean values
        df_mean = df[numeric_columns].mean().to_frame(name="Average").reset_index()
        df_mean.rename(columns={"index": "Metric"}, inplace=True)

        return df_mean


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

    def find_best_run(self, data_list):
        mode = "sort"

        if      mode == "fun":      return self.find_best_run___fun(data_list)
        elif    mode == "sort":     return self.find_best_run___sort(data_list)
        else:   return None


    def find_best_run___sort(self, data_list):
        """
        Finds the best run based on a trade-off between epoch count and loss values.
        """
        if not data_list:
            return None

        weights_1 = {
            "test_loss": 0.3,
            "train_loss": 0.2,
            "val_loss": 0.2,
            "best_epoch": 0.1,
            "time": 0.2
        }

        weights_2 = {
            "test_loss": 0.6,  # Test loss is the most important.
            "train_loss": 0.1,  # Train loss is less critical.
            "val_loss": 0.2,  # Validation loss is still important.
            "best_epoch": 0.1  # Epoch count has low priority.
        }


        # Function to compute a score for each run
        weights = weights_1
        def score(run):
            return (
                run["test_loss"] * weights["test_loss"] +
                run["train_loss"] * weights["train_loss"] +
                run["val_loss"] * weights["val_loss"] +
                run["best_epoch"] * weights["best_epoch"] +
                run["time"] * weights["time"]
            )

        # Find the run with the lowest score
        best_run = min(data_list, key=score)

        return best_run


    def find_best_run___fun(self, data_list, tradeoff_function=None):
        """
        Finds the best run based on an external trade-off function.
        
        :param data_list: List of runs.
        :param tradeoff_function: A function that takes a run dictionary and returns a score.
        """
        if not data_list:
            return None

        # Default sorting (same as before)
        if tradeoff_function is None:
            tradeoff_function__1 = lambda x: (x["test_loss"], x["train_loss"], x["val_loss"], x["best_epoch"])
            tradeoff_function__2 = lambda run: (run["test_loss"] * 0.6 + run["val_loss"] * 0.3 + run["train_loss"] * 0.1) * (1 + run["best_epoch"] / 100)

        # Find the best run based on the custom function
        best_run = min(data_list, key=tradeoff_function__2)

        return best_run

    def my_tradeoff(run):
        return (run["test_loss"] * 0.6 + run["val_loss"] * 0.3 + run["train_loss"] * 0.1) * (1 + run["best_epoch"] / 100)



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

    def display_results(self, dataset_name, data_list):
        if not data_list:
            print("No data available")
            return

        df = pd.DataFrame(data_list)

        numeric_columns = self.metrics

        df_mean = df[numeric_columns].mean().to_frame(name="Average").reset_index()
        df_mean.rename(columns={"index": "Metric"}, inplace=True)

        # Find best runs
        best_run___sort = self.find_best_run___sort(data_list)
        best_run___fun = self.find_best_run___fun(data_list)

        if best_run___sort is None or best_run___fun is None:
            print("Error: No best run found.")
            return

        # Convert best runs to DataFrame (Ensure to keep only relevant columns)
        best_run1_df = pd.DataFrame.from_dict(best_run___sort, orient='index', columns=["Best Run (sort)"]).reset_index()
        best_run1_df.rename(columns={"index": "Metric"}, inplace=True)

        best_run2_df = pd.DataFrame.from_dict(best_run___fun, orient='index', columns=["Best Run (fun)"]).reset_index()
        best_run2_df.rename(columns={"index": "Metric"}, inplace=True)

        # Remove unwanted columns (e.g., '_runtime', '_wandb')
        best_run1_df = best_run1_df[best_run1_df["Metric"].isin(numeric_columns)]
        best_run2_df = best_run2_df[best_run2_df["Metric"].isin(numeric_columns)]

        # Merge data while keeping a single "Metric" column
        merged_df = pd.merge(df_mean, best_run1_df, on="Metric", how="left")
        merged_df = pd.merge(merged_df, best_run2_df, on="Metric", how="left")

        # Add Run Name Row
        run_name_row = pd.DataFrame({"Metric": ["Run Name"], 
                                    "Average": [""], 
                                    "Best Run (sort)": [best_run___sort.get('run', 'N/A')], 
                                    "Best Run (fun)": [best_run___fun.get('run', 'N/A')]})

        # Concatenate final dataframe
        merged_df = pd.concat([run_name_row, merged_df], ignore_index=True)

        # Format numbers to 6 decimal places but remove unnecessary trailing zeros
        def format_number(x):
            if isinstance(x, (int, float)):
                return f"{x:.6f}".rstrip('0').rstrip('.')  # Removes trailing zeros
            return x  # Keep non-numeric values as they are

        merged_df.iloc[:, 1:] = merged_df.iloc[:, 1:].applymap(format_number)

        # Check column count before plotting
        if merged_df.shape[1] != 4:
            print("Error: Merged DataFrame does not have 4 columns. Found:", merged_df.shape[1])
            print(merged_df.head())
            return

        # Plot Table
        fig, ax = plt.subplots(figsize=(8, 5))  # Make figure slightly larger
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(cellText=merged_df.values, 
                        colLabels=["Metric", "Average", "Best Run (sort)", "Best Run (fun)"], 
                        cellLoc="center", loc="center")

        # Apply styling
        for (i, key), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(color="red", fontweight="bold")  # Make headers red
            elif key == 0:  # First column
                cell.set_text_props(color="blue", fontweight="bold")  # Make first column blue

        plt.title(f"Average Metrics for {dataset_name}", fontsize=12, fontweight="bold")
        plt.show()


    def display_final_comparison__best(self):
        """Displays a final table comparing the best models across all datasets, highlighting min values in green."""
        if not self.datasets:
            print("No datasets available.")
            return

        numeric_columns = self.metrics

        # Find best model for each dataset
        best_models = {name: self.find_best_run(data) for name, data in self.datasets.items()}

        # Extract numeric values for each dataset
        comparison_data = {dataset: [best_models[dataset].get(metric, "N/A") for metric in numeric_columns]
                        for dataset in self.datasets}

        # Convert to DataFrame (rows = metrics, columns = datasets)
        comparison_df = pd.DataFrame(comparison_data, index=numeric_columns)

        # Add "Run Name" row at the top (Do NOT include it in min search)
        run_names = {dataset: best_models[dataset].get("run", "N/A") for dataset in self.datasets}
        run_name_df = pd.DataFrame(run_names, index=["Run Name"])
        final_df = pd.concat([run_name_df, comparison_df])

        # Convert all numeric values back to floats for proper min calculations
        for col in final_df.columns:
            final_df[col] = final_df[col].apply(lambda x: float(x) if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit() else x)

        # Format numbers correctly (rounding to 6 decimals)
        for col in final_df.columns:
            final_df[col] = final_df[col].apply(lambda x: f"{x:.6f}".rstrip('0').rstrip('.') if isinstance(x, (int, float)) else x)

        # Plot Table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(cellText=final_df.values,
                        colLabels=final_df.columns,
                        rowLabels=final_df.index,
                        cellLoc="center", loc="center")

        # Apply styling
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row (datasets)
                cell.set_text_props(color="black", fontweight="bold")
            elif j == -1:  # First column (metric names)
                cell.set_text_props(color="black", fontweight="bold")

        # Identify min values and color them green (Ignore "Run Name" row)
        for i in range(1, len(final_df)):  # Start from 1 to skip "Run Name"
            numeric_values = {}
            for j, col in enumerate(final_df.columns):
                try:
                    num_value = float(final_df.iloc[i, j])  # Convert to float
                    numeric_values[j] = num_value  # Store valid numeric values
                except ValueError:
                    continue  # Ignore non-numeric values

            if numeric_values:  # Ensure there's at least one valid number
                min_value = min(numeric_values.values())  # Find the min value
                max_value = max(numeric_values.values())

                for j, value in numeric_values.items():
                    if value == min_value:
                        table[i+1, j].set_text_props(color="green", fontweight="bold")  # Highlight in green
                    if value == max_value: 
                        table[i+1, j].set_text_props(color="red", fontweight="bold")

        plt.title("Comparison of Best Models Across Datasets", fontsize=12, fontweight="bold")
        plt.show()


    
    def display_final_comparison__mean(self):
            if not self.datasets:
                print("No datasets available.")
                return

            numeric_columns = self.metrics
            avg_models = {name: self.calculate_average_metrics__new(data) for name, data in self.datasets.items()}
            comparison_data = {dataset: [avg_models[dataset].get(metric, "N/A") for metric in numeric_columns] for dataset in self.datasets}
            comparison_df = pd.DataFrame(comparison_data, index=numeric_columns)

            for col in comparison_df.columns:
                comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
            
            final_df = comparison_df.map(lambda x: f"{x:.6f}".rstrip('0').rstrip('.') if pd.notna(x) else "N/A")

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis("tight")
            ax.axis("off")
            table = ax.table(cellText=final_df.values,
                            colLabels=final_df.columns,
                            rowLabels=final_df.index,
                            cellLoc="center", loc="center")

            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_text_props(color="black", fontweight="bold")
                elif j == -1:
                    cell.set_text_props(color="black", fontweight="bold")

            for i in range(len(final_df)):
                numeric_values = {}
                for j, col in enumerate(final_df.columns):
                    try:
                        num_value = float(final_df.iloc[i, j])
                        numeric_values[j] = num_value
                    except ValueError:
                        continue

                if numeric_values:
                    min_value = min(numeric_values.values())
                    max_value = max(numeric_values.values())

                    for j, value in numeric_values.items():
                        if value == min_value:
                            table[i+1, j].set_text_props(color="green", fontweight="bold")
                        if value == max_value:
                            table[i+1, j].set_text_props(color="red", fontweight="bold")

            plt.title("Comparison of Average Models Across Datasets", fontsize=12, fontweight="bold")
            plt.show()

    def calculate_average_metrics__new(self, data_list):
        if not data_list:
            return None
        df = pd.DataFrame(data_list)
        numeric_columns = [col for col in self.metrics if col in df.columns]
        return df[numeric_columns].mean().to_dict()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

    def plot_metric_across_datasets(self, datasets, metric):
        """
        Plots the specified metric for each dataset.
        
        :param datasets: Dictionary of datasets, where each dataset contains a list of runs.
        :param metric: The metric to plot (default: "val_loss").
        """
        plt.figure(figsize=(10, 6))

        for dataset_name, data_list in datasets.items():
            if not data_list:
                continue

            # Convert data to DataFrame
            df = pd.DataFrame(data_list)

            if metric not in df.columns:
                print(f"Metric '{metric}' not found in dataset '{dataset_name}'. Skipping.")
                continue

            # Extract _step values (or epoch, if step is missing) and the chosen metric
            df_sorted = df.sort_values(by="_step")
            x_values = df_sorted["_step"] if "_step" in df_sorted else df_sorted["epoch"]
            y_values = df_sorted[metric]

            # Plot each dataset's metric trajectory
            plt.plot(x_values, y_values, marker='o', linestyle='-', label=dataset_name)

        plt.xlabel("Training Steps (or Epochs)")
        plt.ylabel(metric.replace("_", " ").capitalize())  # Formatting metric name for display
        plt.title(f"Comparison of {metric} Across Datasets")
        plt.legend()
        plt.grid(True)
        plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

CalcValues()