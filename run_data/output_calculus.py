import pandas as pd
import matplotlib.pyplot as plt
import argparse
import csv, ast, itertools


class CalcValues:
    def __init__(self, display=False, plot=False, all=False, loc=None):
        self.datasets = self.DataSet(all) if loc is None else self.get_from_csv(loc)
        #print(self.datasets)

        self.metrics = [
            "_step", 
            #"best_epoch", 
            "best_test", 
            "best_train", 
            "best_val", 
            #"lr", 
            #"test_loss", 
            "time", 
            #"train_loss", 
            #"val_loss"
            ]

        if display:
            for name, data in self.datasets.items():    self.display_results(name, data)
            if loc is None:                             self.display_final_comparison__best()
            if loc is None:                             self.display_final_comparison__mean()

        if plot:
            for param in ["best_val", "best_test", "epoch"]: 
                self.plot_metric_across_datasets(self.datasets, param)


    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Run results analysis with optional display and plotting.")
        parser.add_argument("--display", action="store_true", help="Display results and comparisons.")
        parser.add_argument("--plot", action="store_true", help="Plot selected metrics across datasets.")
        parser.add_argument("--all", action="store_true", help="Select all metrics across datasets.")
        parser.add_argument("--loc", type=str, help="Path to the directory")
        args = parser.parse_args()
        return args


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

    def DataSet(self, all):
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

        improving = [
            {'run': "034(b)", '_runtime': 21600.8713907, '_step': 125, '_timestamp': 1741641582.1955037, '_wandb': {'runtime': 21600}, 'best_epoch': 106, 'best_test': 0.1035991437910568, 'best_train': 0.01831837007332416, 'best_val': 0.02983867224778921, 'epoch': 126, 'lr': 0.000125, 'test_loss': 0.10504141532712512, 'time': 21569.769416570663, 'train_loss': 0.0167576711299637, 'val_loss': 0.03204075362355936}
            , {'run': "035(a)", '_runtime': 17827.271294305, '_step': 130, '_timestamp': 1741641105.369735, '_wandb': {'runtime': 17827}, 'best_epoch': 111, 'best_test': 0.0369229315233136, 'best_train': 0.01782039366169739, 'best_val': 0.047414959925744266, 'epoch': 131, 'lr': 0.000125, 'test_loss': 0.03637387721784531, 'time': 17803.01145029068, 'train_loss': 0.01787952652999333, 'val_loss': 0.05475342119970019}
            , {'run': "035(a)", '_runtime': 20161.1053874, '_step': 126, '_timestamp': 1741669341.200771, '_wandb': {'runtime': 20161}, 'best_epoch': 107, 'best_test': 0.15756188126073944, 'best_train': 0.025609337325607027, 'best_val': 0.227717852545163, 'epoch': 127, 'lr': 0.00025, 'test_loss': 0.13595869276849998, 'time': 20122.635061740875, 'train_loss': 0.024088118387947954, 'val_loss': 0.22873365618879832}
            , {'run': "037(b)", '_runtime': 23429.738305802, '_step': 183, '_timestamp': 1741672659.4781094, '_wandb': {'runtime': 23429}, 'best_epoch': 164, 'best_test': 0.05146576198084014, 'best_train': 0.01903745942507629, 'best_val': 0.02815484595558946, 'epoch': 184, 'lr': 0.000125, 'test_loss': 0.0547664255968162, 'time': 23409.78754425049, 'train_loss': 0.018740728216630125, 'val_loss': 0.028786422803051888}
        ]

        sweep = [
            {'run': "swift-sweep-1", '_runtime': 18845.1666517, '_step': 173, '_timestamp': 1741729763.481567, '_wandb': {'runtime': 18845}, 'batch_size': 256, 'best_epoch': 153, 'best_test': 0.1481804974609986, 'best_train': 0.014840607933307949, 'best_val': 0.02342821948695928, 'epoch': 173, 'lr': 0.000125, 'n_epochs': 500, 'test_loss': 0.15058815176598728, 'time': 18822.825367212296, 'train_loss': 0.013035300756363492, 'val_loss': 0.02358409221051261}
            , {'run': "floral-sweep-3", '_runtime': 29706.1014717, '_step': 177, '_timestamp': 1741759500.771211, '_wandb': {'runtime': 29706}, 'batch_size': 256, 'best_epoch': 167, 'best_test': 0.29612932493910193, 'best_train': 0.19118200509171737, 'best_val': 0.19001176580786705, 'epoch': 177, 'lr': 0.0005, 'n_epochs': 500, 'test_loss': 0.44739476684480906, 'time': 29688.96263360977, 'train_loss': 0.3677213778621272, 'val_loss': 0.38294084928929806}
            , {'run': "glamorous-sweep-1", '_runtime': 12708.964212641, '_step': 91, '_timestamp': 1741747415.9264305, '_wandb': {'runtime': 12708}, 'batch_size': 128, 'best_epoch': 71, 'best_test': 0.07949973113598331, 'best_train': 0.0155695670791877, 'best_val': 0.05276760600862049, 'epoch': 91, 'lr': 0.00025, 'n_epochs': 1000, 'test_loss': 0.08527710055193258, 'time': 12689.04225039482, 'train_loss': 0.0128133891731816, 'val_loss': 0.07545831897074268}
            , {'run': "charmed-sweep-2", '_runtime': 8683.298607489, '_step': 32, '_timestamp': 1741756105.3313603, '_wandb': {'runtime': 8683}, 'batch_size': 64, 'best_epoch': 22, 'best_test': 0.08491663201638157, 'best_train': 0.019425083574312753, 'best_val': 0.06633073543863637, 'epoch': 32, 'lr': 0.001, 'n_epochs': 500, 'test_loss': 0.09113348185247372, 'time': 8653.249694108963, 'train_loss': 0.025678014014637596, 'val_loss': 0.0816020660457157}
            , {'run': "fluent-sweep-3", '_runtime': 9001.820236002, '_step': 62, '_timestamp': 1741765112.1062276, '_wandb': {'runtime': 9001}, 'batch_size': 128, 'best_epoch': 52, 'best_test': 0.08140742784691235, 'best_train': 0.023643729350869617, 'best_val': 0.07006013795496925, 'epoch': 62, 'lr': 0.0005, 'n_epochs': 1000, 'test_loss': 0.1155524601538976, 'time': 8980.800916910172, 'train_loss': 0.03282010080736308, 'val_loss': 0.10186874286995994}            
            , {'run': "radiant-sweep-4", '_runtime': 23897.1032826, '_step': 88, '_timestamp': 1741783404.1021, '_wandb': {'runtime': 23897}, 'batch_size': 128, 'best_epoch': 69, 'best_test': 0.11909919634224876, 'best_train': 0.012003492137722713, 'best_val': 0.019937452165380357, 'epoch': 88, 'lr': 0.000125, 'n_epochs': 500, 'test_loss': 0.12107125542584866, 'time': 23859.6831073761, 'train_loss': 0.011088666082335212, 'val_loss': 0.02143756610651811}           
            , {'run': "faithful-sweep-4", '_runtime': 4399.594476623, '_step': 30, '_timestamp': 1741769523.377036, '_wandb': {'runtime': 4399}, 'batch_size': 128, 'best_epoch': 10, 'best_test': 0.12021032513843642, 'best_train': 0.03149817268999796, 'best_val': 0.08299240676893128, 'epoch': 30, 'lr': 0.001, 'n_epochs': 1000, 'test_loss': 0.10709275855194957, 'time': 4381.372041940689, 'train_loss': 0.013504245855584346, 'val_loss': 0.0952433996966907}            
            , {'run': "lively-sweep-5", '_runtime': 9514.573641236, '_step': 35, '_timestamp': 1741779042.769012, '_wandb': {'runtime': 9514}, 'batch_size': 64, 'best_epoch': 15, 'best_test': 0.17331557913077256, 'best_train': 0.030269269110135912, 'best_val': 0.15547048885907447, 'epoch': 35, 'lr': 0.001, 'n_epochs': 500, 'test_loss': 0.13399746554297587, 'time': 9484.192527770996, 'train_loss': 0.011309588657795829, 'val_loss': 0.2246868521389034}           
            , {'run': "sleek-sweep-6", '_runtime': 4464.873752629, '_step': 51, '_timestamp': 1741783515.3837569, '_wandb': {'runtime': 4464}, 'batch_size': 256, 'best_epoch': 50, 'best_test': 0.3749983357265592, 'best_train': 0.2596939421013782, 'best_val': 0.4621657971292734, 'epoch': 51, 'lr': 0.0001, 'n_epochs': 500, 'test_loss': 0.3714748676866293, 'time': 4450.12158370018, 'train_loss': 0.25333702642666667, 'val_loss': 0.4650606904178858}          
            , {'run': "stellar-sweep-6", '_runtime': 8325.8550323, '_step': 73, '_timestamp': 1741792001.879764, '_wandb': {'runtime': 8325}, 'batch_size': 256, 'best_epoch': 63, 'best_test': 0.3030208139680326, 'best_train': 0.17432295476135454, 'best_val': 0.19699118053540587, 'epoch': 73, 'lr': 0.002, 'n_epochs': 500, 'test_loss': 0.3361325887963176, 'time': 8306.17468380928, 'train_loss': 0.17055228681940782, 'val_loss': 0.22658801125362515}
            , {'run': "gallant-sweep-9", '_runtime': 12043.8551942, '_step': 109, '_timestamp': 1741804125.6145654, '_wandb': {'runtime': 12043}, 'batch_size': 256, 'best_epoch': 109, 'best_test': 0.38882978167384863, 'best_train': 0.306771037296245, 'best_val': 0.369750595651567, 'epoch': 109, 'lr': 0.0001, 'n_epochs': 1000, 'test_loss': 0.38882978167384863, 'time': 12019.04819726944, 'train_loss': 0.306771037296245, 'val_loss': 0.369750595651567}     
            , {'run': "solar-sweep-1", '_runtime': 22371.873769663, '_step': 259, '_timestamp': 1741920591.2233338, '_wandb': {'runtime': 22371}, 'batch_size': 256, 'best_epoch': 249, 'best_test': 0.2518298481591046, 'best_train': 0.11173687953698008, 'best_val': 0.1492697079665959, 'epoch': 259, 'lr': 0.0005, 'n_epochs': 500, 'test_loss': 0.2549635097384453, 'time': 22352.28443312645, 'train_loss': 0.1307731625280882, 'val_loss': 0.1593626686371863}
            , {'run': "hearty-sweep-2", '_runtime': 25416.542699058, '_step': 158, '_timestamp': 1741946020.719251, '_wandb': {'runtime': 25416}, 'batch_size': 128, 'best_epoch': 138, 'best_test': 0.320854652968664, 'best_train': 0.010838985985154828, 'best_val': 0.09456262898646176, 'epoch': 158, 'lr': 0.000125, 'n_epochs': 500, 'test_loss': 0.3193950263990296, 'time': 25397.20451593399, 'train_loss': 0.007751289033207786, 'val_loss': 0.09689457853516888}
            , {'run': "stoic-sweep-3", '_runtime': 20058.501074448, '_step': 197, '_timestamp': 1741966086.260381, '_wandb': {'runtime': 20058}, 'batch_size': 256, 'best_epoch': 177, 'best_test': 0.2560703344643116, 'best_train': 0.01627925415768435, 'best_val': 0.10055328393355012, 'epoch': 197, 'lr': 0.000125, 'n_epochs': 1000, 'test_loss': 0.25492757512256503, 'time': 20042.328552007675, 'train_loss': 0.011001438534769571, 'val_loss': 0.10400280356407166}
            , {'run': "serene-sweep-4", '_runtime': 22556.832118349, '_step': 243, '_timestamp': 1741988648.1885312, '_wandb': {'runtime': 22594}, 'batch_size': 256, 'best_epoch': 227, 'best_test': 0.288103133905679, 'best_train': 0.021198425412570176, 'best_val': 0.11791448248550296, 'epoch': 243, 'lr': 0.00025, 'n_epochs': 500, 'test_loss': 0.29501010989770293, 'time': 22538.295169591904, 'train_loss': 0.013506667806129706, 'val_loss': 0.12043031281791627}
        ]

        hyperparams = [
            {'run': "038", '_runtime': 18812.5908758, '_step': 113, '_timestamp': 1741824679.444635, '_wandb': {'runtime': 18985}, 'batch_size': 128, 'best_epoch': 98, 'best_test': 0.032758295240383296, 'best_train': 0.018323482102936224, 'best_val': 0.1541849288734652, 'epoch': 113, 'lr': 0.000125, 'n_epochs': 500, 'test_loss': 0.03999957615243537, 'time': 18786.76917457581, 'train_loss': 0.009215078745332976, 'val_loss': 0.1645800969785168}
            , {'run': "039", '_runtime': 13081.172081066, '_step': 64, '_timestamp': 1741819893.582723, '_wandb': {'runtime': 13081}, 'batch_size': 64, 'best_epoch': 54, 'best_test': 0.426040991845112, 'best_train': 0.0412505230784851, 'best_val': 0.0775034295927201, 'epoch': 64, 'lr': 0.00075, 'n_epochs': 600, 'test_loss': 0.3696995698329475, 'time': 13057.94013261795, 'train_loss': 0.022297281211128443, 'val_loss': 0.0940718236157582}
            , {'run': "040", '_runtime': 8827.003309655, '_step': 120, '_timestamp': 1741833651.5183764, '_wandb': {'runtime': 8827}, 'batch_size': 256, 'best_epoch': 100, 'best_test': 0.059200827148742974, 'best_train': 0.029671121996484303, 'best_val': 0.04542305477662012, 'epoch': 120, 'lr': 0.00025, 'n_epochs': 500, 'test_loss': 0.06239888025447726, 'time': 8806.624128103256, 'train_loss': 0.019612554205875647, 'val_loss': 0.05699803400784731}
            , {'run': "041", '_runtime': 23117.1761192, '_step': 144, '_timestamp': 1741848031.0513425, '_wandb': {'runtime': 23118}, 'batch_size': 128, 'best_epoch': 124, 'best_test': 0.0475185111697231, 'best_train': 0.01072292075931001, 'best_val': 0.10124371233322316, 'epoch': 144, 'lr': 6.25e-05, 'n_epochs': 500, 'test_loss': 0.0479222886146061, 'time': 23089.85883140564, 'train_loss': 0.009571122266746388, 'val_loss': 0.1117571032946072}
            , {'run': "042", '_runtime': 18601.3108741, '_step': 87, '_timestamp': 1741882667.967213, '_wandb': {'runtime': 18601}, 'batch_size': 128, 'best_epoch': 67, 'best_test': 0.06634693616439426, 'best_train': 0.031640093990419275, 'best_val': 0.09006652643992788, 'epoch': 87, 'lr': 0.0005, 'n_epochs': 1000, 'test_loss': 0.05605158446327088, 'time': 18531.320506334305, 'train_loss': 0.020195376044188543, 'val_loss': 0.09410358344515164}
            , {'run': "043", '_runtime': 7603.51320808, '_step': 58, '_timestamp': 1741873667.2091389, '_wandb': {'runtime': 7603}, 'batch_size': 128, 'best_epoch': 38, 'best_test': 0.04849234371194764, 'best_train': 0.018132071237439516, 'best_val': 0.052082264322846655, 'epoch': 58, 'lr': 0.00025, 'n_epochs': 500, 'test_loss': 0.04370142495821393, 'time': 7582.564762830734, 'train_loss': 0.014893760269005149, 'val_loss': 0.05963793784261696}
            , {'run': "044", '_runtime': 6459.454143329, '_step': 93, '_timestamp': 1741882446.3778567, '_wandb': {'runtime': 6459}, 'batch_size': 256, 'best_epoch': 73, 'best_test': 0.0933970334008336, 'best_train': 0.046414570353533094, 'best_val': 0.10525911825243384, 'epoch': 93, 'lr': 0.0005, 'n_epochs': 1000, 'test_loss': 0.100127711892128, 'time': 6442.120647192001, 'train_loss': 0.03298328363973843, 'val_loss': 0.2127576768398285}
            , {'run': "045", '_runtime': 11935.6086048, '_step': 72, '_timestamp': 1741898373.9301524, '_wandb': {'runtime': 11935}, 'batch_size': 128, 'best_epoch': 37, 'best_test': 0.05453523044430074, 'best_train': 0.02867371845675051, 'best_val': 0.0871459778457407, 'epoch': 72, 'lr': 0.00025, 'n_epochs': 1000, 'test_loss': 0.05989702445055757, 'time': 11906.330340385435, 'train_loss': 0.01179708170620774, 'val_loss': 0.11624310623913532}
            , {'run': "046", '_runtime': 7944.6774927, '_step': 65, '_timestamp': 1741894454.9260068, '_wandb': {'runtime': 7944}, 'batch_size': 128, 'best_epoch': 45, 'best_test': 0.057132279145575705, 'best_train': 0.03496928833346203, 'best_val': 0.1002613367542388, 'epoch': 65, 'lr': 0.0005, 'n_epochs': 1000, 'test_loss': 0.05603789833803025, 'time': 7925.196460485458, 'train_loss': 0.026033652389530468, 'val_loss': 0.12433141328039624}
            , {'run': "047", '_runtime': 15373.7449627, '_step': 61, '_timestamp': 1742229683.6684904, '_wandb': {'runtime': 15373}, 'batch_size': 128, 'best_epoch': 57, 'best_test': 0.03277164156593028, 'best_train': 0.005900937812550674, 'best_val': 0.025855697661874785, 'epoch': 61, 'lr': 0.0005, 'n_epochs': 600, 'test_loss': 0.03224825329842076, 'time': 15343.459737062454, 'train_loss': 0.0073168578018094335, 'val_loss': 0.028925311145564868}
            , {'run': "048", '_runtime': 13606.6700094, '_step': 58, '_timestamp': 1742249642.4426694, '_wandb': {'runtime': 13621}, 'batch_size': 128, 'best_epoch': 55, 'best_test': 0.09819107535221272, 'best_train': 0.021722012854836607, 'best_val': 0.04362508317544347, 'coeff_train': 0.6076177960802271, 'epoch': 58, 'lr': 0.0005, 'n_epochs': 200, 'test_loss': 0.08230823340515296, 'time': 13542.980422735214, 'train_loss': 0.022515423196767057, 'val_loss': 0.08573220776660102}
            , {'run': "049", '_runtime': 11995.576234156, '_step': 72, '_timestamp': 1742249640.9203196, '_wandb': {'runtime': 12024}, 'batch_size': 128, 'best_epoch': 68, 'best_test': 0.2817918884139212, 'best_train': 0.010568339875332577, 'best_val': 0.05043999835967072, 'coeff_train': 0.5078245317955713, 'epoch': 72, 'lr': 0.0005, 'n_epochs': 200, 'test_loss': 0.2890660311021502, 'time': 11965.97043466568, 'train_loss': 0.010115224222539278, 'val_loss': 0.05292532734927677}
        ]

        sweep_test1 = [
            {'run': "misty-sweep-1", '_runtime': 11024.115094132, '_step': 120, '_timestamp': 1742332662.2408974, '_wandb': {'runtime': 11024}, 'batch_size': 256, 'best_epoch': 100, 'best_test': 0.06888132111635059, 'best_train': 0.019719987686135268, 'best_val': 0.1474378340644762, 'coeff_train': -0.0029620062603088557, 'epoch': 120, 'lr': 0.00025, 'n_epochs': 250, 'test_loss': 0.061660713981837034, 'time': 11005.445610761642, 'train_loss': 0.01480889403702397, 'val_loss': 0.15918675158172846}
            , {'run': "tough-sweep-2", '_runtime': 9072.666214127, '_step': 98, '_timestamp': 1742341739.210579, '_wandb': {'runtime': 9072}, 'batch_size': 256, 'best_epoch': 78, 'best_test': 0.06129547767341137, 'best_train': 0.019257749783757487, 'best_val': 0.1410989115247503, 'coeff_train': 0.28187683257297497, 'epoch': 98, 'lr': 0.00025, 'n_epochs': 250, 'test_loss': 0.06198051362298429, 'time': 9055.438225507736, 'train_loss': 0.0172425540262147, 'val_loss': 0.1520931125851348}
            , {'run': "faithful-sweep-3", '_runtime': 12593.076943028, '_step': 137, '_timestamp': 1742354339.667109, '_wandb': {'runtime': 12593}, 'batch_size': 256, 'best_epoch': 117, 'best_test': 0.058772760326974094, 'best_train': 0.01931111704754202, 'best_val': 0.1228409290779382, 'coeff_train': 0.1475546443047215, 'epoch': 137, 'lr': 3.125e-05, 'n_epochs': 250, 'test_loss': 0.0591220295173116, 'time': 12575.965531110764, 'train_loss': 0.01811363609801782, 'val_loss': 0.12682463508099318}
            , {'run': "magic-sweep-4", '_runtime': 2969.878919974, '_step': 32, '_timestamp': 1742357315.3178823, '_wandb': {'runtime': 2969}, 'batch_size': 256, 'best_epoch': 31, 'best_test': 0.07250649214256555, 'best_train': 0.05460579689396055, 'best_val': 0.2550557474605739, 'coeff_train': -0.02147659520011636, 'epoch': 32, 'lr': 0.0005, 'n_epochs': 250, 'test_loss': 0.07228638930246234, 'time': 2952.6616582870483, 'train_loss': 0.058277693782982073, 'val_loss': 0.2611319711431861}
        ]

        sweep_test2 = [
            {'run': "confused-sweep-1", '_runtime': 51452.0871214, '_step': 250, '_timestamp': 1742372941.412101, '_wandb': {'runtime': 51452}, 'batch_size': 128, 'best_epoch': 239, 'best_test': 0.031004800801239317, 'best_train': 0.009287979094084924, 'best_val': 0.05454751444123094, 'coeff_train': 0.48955478367718497, 'epoch': 250, 'lr': 7.8125e-06, 'n_epochs': 250, 'test_loss': 0.031127572089197145, 'time': 51363.07405257225, 'train_loss': 0.009128478685857129, 'val_loss': 0.055135754779690786}
            , {'run': "kind-sweep-2", '_runtime': 8845.4009014, '_step': 40, '_timestamp': 1742381794.2783391, '_wandb': {'runtime': 9179}, 'batch_size': 128, 'best_epoch': 39, 'best_test': 0.033578491636684964, 'best_train': 0.018692423402277565, 'best_val': 0.10012891486523644, 'coeff_train': 0.5069728789637931, 'epoch': 40, 'lr': 0.0005, 'n_epochs': 250, 'test_loss': 0.03411439492825478, 'time': 8819.940562725067, 'train_loss': 0.02000996027457178, 'val_loss': 0.1017378441516369}
            , {'run': "wandering-sweep-1", '_runtime': 22879.018983845, '_step': 140, '_timestamp': 1742395134.0447586, '_wandb': {'runtime': 22879}, 'batch_size': 128, 'best_epoch': 120, 'best_test': 0.034756482770991706, 'best_train': 0.013779155131449145, 'best_val': 0.03721843429264568, 'coeff_train': 0.5139751889812413, 'epoch': 140, 'lr': 1.5625e-05, 'n_epochs': 250, 'test_loss': 0.03257751355450305, 'time': 22859.748757839203, 'train_loss': 0.012557587495182085, 'val_loss': 0.03811853015351863}
            , {'run': "cosmic-sweep-2", '_runtime': 15593.099980067, '_step': 95, '_timestamp': 1742410733.3457065, '_wandb': {'runtime': 15593}, 'batch_size': 128, 'best_epoch': 75, 'best_test': 0.035720562266688495, 'best_train': 0.014536536106514552, 'best_val': 0.021167820777803187, 'coeff_train': 0.4582362064279332, 'epoch': 95, 'lr': 0.000125, 'n_epochs': 250, 'test_loss': 0.03200252268404242, 'time': 15573.552463769913, 'train_loss': 0.011725604201533964, 'val_loss': 0.02607648769423129}
            , {'run': "ethereal-sweep-3", '_runtime': 15805.703061435, '_step': 100, '_timestamp': 1742426551.79379, '_wandb': {'runtime': 15805}, 'batch_size': 128, 'best_epoch': 80, 'best_test': 0.03534930095904403, 'best_train': 0.015882374994732714, 'best_val': 0.04173140087357116, 'coeff_train': 0.4843090560305771, 'epoch': 100, 'lr': 0.00025, 'n_epochs': 250, 'test_loss': 0.030755848224673952, 'time': 15786.911224365234, 'train_loss': 0.013034136002518673, 'val_loss': 0.04548661978471847}
            , {'run': "warm-sweep-4", '_runtime': 10038.341858196, '_step': 64, '_timestamp': 1742436594.9334784, '_wandb': {'runtime': 10038}, 'batch_size': 128, 'best_epoch': 44, 'best_test': 0.034704865887761116, 'best_train': 0.018478659701055635, 'best_val': 0.022942104214240634, 'coeff_train': 0.4962330592014699, 'epoch': 64, 'lr': 0.00025, 'n_epochs': 250, 'test_loss': 0.03680158755372441, 'time': 10019.349866390228, 'train_loss': 0.028050923685508745, 'val_loss': 0.03192913694868958}
            , {'run': "dutiful-sweep-5", '_runtime': 17242.568128226, '_step': 110, '_timestamp': 1742453844.7123826, '_wandb': {'runtime': 17242}, 'batch_size': 128, 'best_epoch': 90, 'best_test': 0.030302857152289815, 'best_train': 0.015441546305304482, 'best_val': 0.05179509887146571, 'coeff_train': 0.5215770454153691, 'epoch': 110, 'lr': 0.00025, 'n_epochs': 250, 'test_loss': 0.029026392433378432, 'time': 17224.029055833817, 'train_loss': 0.015409179423595704, 'val_loss': 0.05471389678617319}
            , {'run': "noble-sweep-6", '_runtime': 19138.104573753, '_step': 122, '_timestamp': 1742472987.487784, '_wandb': {'runtime': 19138}, 'batch_size': 128, 'best_epoch': 102, 'best_test': 0.03758969905948828, 'best_train': 0.01295366488081753, 'best_val': 0.05060393246452487, 'coeff_train': 0.4709107394242168, 'epoch': 122, 'lr': 0.000125, 'n_epochs': 250, 'test_loss': 0.03624591946838394, 'time': 19119.55045223236, 'train_loss': 0.01371896853079162, 'val_loss': 0.05073051462097773}
            , {'run': "rural-sweep-7", '_runtime': 21334.1969987, '_step': 136, '_timestamp': 1742494325.6185322, '_wandb': {'runtime': 21334}, 'batch_size': 128, 'best_epoch': 116, 'best_test': 0.04271604469607747, 'best_train': 0.013630910305966618, 'best_val': 0.06053380789383063, 'coeff_train': 0.469988073124419, 'epoch': 136, 'lr': 6.25e-05, 'n_epochs': 250, 'test_loss': 0.03963977691259176, 'time': 21315.650056362152, 'train_loss': 0.011772438460005023, 'val_loss': 0.0624421988687818}
            , {'run': "fanciful-sweep-8", '_runtime': 15216.210767826, '_step': 97, '_timestamp': 1742509548.5442338, '_wandb': {'runtime': 15216}, 'batch_size': 128, 'best_epoch': 77, 'best_test': 0.034048171093066536, 'best_train': 0.016180047664357714, 'best_val': 0.03292425072914551, 'coeff_train': 0.4756777258852426, 'epoch': 97, 'lr': 0.00025, 'n_epochs': 250, 'test_loss': 0.03009101931774427, 'time': 15197.702519655228, 'train_loss': 0.012170023903270403, 'val_loss': 0.03428307033720471}
            , {'run': "rosy-sweep-9", '_runtime': 12243.511254699, '_step': 78, '_timestamp': 1742521799.144714, '_wandb': {'runtime': 12243}, 'batch_size': 128, 'best_epoch': 58, 'best_test': 0.026237598428177456, 'best_train': 0.021953875460124835, 'best_val': 0.02995680191273254, 'coeff_train': 0.5100325364576248, 'epoch': 78, 'lr': 0.00025, 'n_epochs': 250, 'test_loss': 0.02548247750198084, 'time': 12225.03886294365, 'train_loss': 0.0153129525758602, 'val_loss': 0.0347361501365427}
            , {'run': "fine-sweep-10", '_runtime': 25608.000590301, '_step': 163, '_timestamp': 1742547414.9314828, '_wandb': {'runtime': 25664}, 'batch_size': 128, 'best_epoch': 150, 'best_test': 0.032055843118873856, 'best_train': 0.012655384966699537, 'best_val': 0.033084150980271045, 'coeff_train': 0.5192063002641719, 'epoch': 163, 'lr': 3.125e-05, 'n_epochs': 250, 'test_loss': 0.03175701381313422, 'time': 25589.49342560768, 'train_loss': 0.011259140104311642, 'val_loss': 0.034750476568227724}
            , {'run': "unique-sweep-1___gru-correct", '_runtime': 3160.482578693, '_step': 39, '_timestamp': 1742568775.206573, '_wandb': {'runtime': 3161}, 'batch_size': 256, 'best_epoch': 19, 'best_test': 0.2146770553663373, 'best_train': 0.03612650911274709, 'best_val': 0.25488561973907053, 'coeff_train': 0.3750022198628035, 'epoch': 39, 'lr': 0.00025, 'n_epochs': 250, 'test_loss': 0.18053001631051305, 'time': 3142.2687587738037, 'train_loss': 0.019084395997618377, 'val_loss': 0.2763296226039529}
            , {'run': "fanciful-sweep-2___new-correct", '_runtime': 9720.829009913, '_step': 102, '_timestamp': 1742578506.207727, '_wandb': {'runtime': 9722}, 'batch_size': 256, 'best_epoch': 82, 'best_test': 0.15718137589283288, 'best_train': 0.02246262655642472, 'best_val': 0.24477599072270095, 'coeff_train': 0.3367758074012364, 'epoch': 102, 'lr': 0.000125, 'n_epochs': 250, 'test_loss': 0.1465441077016294, 'time': 9704.493029117584, 'train_loss': 0.017321900297936642, 'val_loss': 0.24485561018809676}
            , {'run': "faithful-sweep-3___new-correct", '_runtime': 4761.038547686, '_step': 50, '_timestamp': 1742583282.1184676, '_wandb': {'runtime': 4762}, 'batch_size': 256, 'best_epoch': 30, 'best_test': 0.2010010532103479, 'best_train': 0.07917240022828705, 'best_val': 0.26498157205060124, 'coeff_train': 0.3815396484450914, 'epoch': 50, 'lr': 0.00025, 'n_epochs': 250, 'test_loss': 0.1529308415483683, 'time': 4744.61030626297, 'train_loss': 0.03101953019044901, 'val_loss': 0.2867288014385849}
            , {'run': "blooming-sweep-4___gru-correct", '_runtime': 419.991751438, '_step': 5, '_timestamp': 1742583710.673082, '_wandb': {'runtime': 419}, 'batch_size': 256, 'best_epoch': 5, 'best_test': 0.4560015583410859, 'best_train': 0.14651843834864467, 'best_val': 0.45643745455890894, 'coeff_train': 0.3906993182737436, 'epoch': 5, 'lr': 0.0005, 'n_epochs': 250, 'test_loss': 0.4560015583410859, 'time': 404.3699789047241, 'train_loss': 0.18132653597154116, 'val_loss': 0.45643745455890894}
        ]

        vdp_gru = [
            {'run': "050___vdp", '_runtime': 37481.6913807, '_step': 139, '_timestamp': 1742460651.8235345, '_wandb': {'runtime': 37754}, 'batch_size': 128, 'best_epoch': 139, 'best_test': 0.03165482283229866, 'best_train': 0.013442921733099316, 'best_val': 0.06372759999736907, 'coeff_train': 0.580947917413158, 'epoch': 139, 'lr': 3.125e-05, 'n_epochs': 500, 'test_loss': 0.03165482283229866, 'time': 37451.43218612671, 'train_loss': 0.013442921733099316, 'val_loss': 0.06372759999736907}
            , {'run': "054___vdp-set3(n.c.)", '_runtime': 9844.692996, '_step': 33, '_timestamp': 1742490344.026744, '_wandb': {'runtime': 10118}, 'batch_size': 128, 'best_epoch': 26, 'best_test': 0.02325438444931356, 'best_train': 0.01138090737479389, 'best_val': 0.04191491572511575, 'coeff_train': 0.4454360099969614, 'epoch': 33, 'lr': 0.001, 'n_epochs': 600, 'test_loss': 0.026860858254607708, 'time': 9747.53641295433, 'train_loss': 0.010035054814858883, 'val_loss': 0.05350164176216201}
            , {'run': "053___vdp-default-code", '_runtime': 9794.1529896, '_step': 117, '_timestamp': 1742466946.665296, '_wandb': {'runtime': 9861}, 'batch_size': 128, 'best_epoch': 112, 'best_test': 0.033960099849436015, 'best_train': 0.013545496137960562, 'best_val': 0.06969134811134566, 'coeff_train': 0, 'epoch': 117, 'lr': 0.000125, 'n_epochs': 500, 'test_loss': 0.03329164023318934, 'time': 9766.92120027542, 'train_loss': 0.013237217680191395, 'val_loss': 0.07226177934734594}
            , {'run': "055___vdp-set3-gru(n.c.)", '_runtime': 7211.1819151, '_step': 28, '_timestamp': 1742490299.8464868, '_wandb': {'runtime': 7520}, 'batch_size': 128, 'best_epoch': 19, 'best_test': 0.03480247207104214, 'best_train': 0.011666294832827238, 'best_val': 0.03446821563891948, 'coeff_train': 0.628063944916227, 'epoch': 28, 'lr': 0.001, 'n_epochs': 600, 'test_loss': 0.03800599498762971, 'time': 7170.803230762482, 'train_loss': 0.008743340018939562, 'val_loss': 0.03799575786032374}
            , {'run': "057___vdp-set5-gru", '_runtime': 19531.4914334, '_step': 123, '_timestamp': 1742521861.9610133, '_wandb': {'runtime': 19533}, 'batch_size': 256, 'best_epoch': 103, 'best_test': 0.04588673071702942, 'best_train': 0.009910718660409512, 'best_val': 0.06170963332988322, 'coeff_train': 0.3947962669195425, 'epoch': 123, 'lr': 0.000125, 'n_epochs': 200, 'test_loss': 0.04987818584777415, 'time': 19489.789879083633, 'train_loss': 0.009450188705599622, 'val_loss': 0.07695418025832623}
        ]

        fhn_model = [
            {'run': "051___fhn", '_runtime': 22108.2997181, '_step': 77, '_timestamp': 1742446219.0934548, '_wandb': {'runtime': 22108}, 'batch_size': 128, 'best_epoch': 57, 'best_test': 0.031174158928768028, 'best_train': 0.028783515184408144, 'best_val': 0.036670097874270544, 'coeff_train': 0.6384884914948572, 'epoch': 77, 'lr': 0.00025, 'n_epochs': 500, 'test_loss': 0.02827900384242336, 'time': 22070.82731485367, 'train_loss': 0.02422400077351581, 'val_loss': 0.04486035347162258}
            , {'run': "052___fhn-default-code", '_runtime': 13967.7160078, '_step': 127, '_timestamp': 1742439179.7070515, '_wandb': {'runtime': 13968}, 'batch_size': 128, 'best_epoch': 107, 'best_test': 0.026796013301622772, 'best_train': 0.022122922061484248, 'best_val': 0.03483917065970008, 'coeff_train': 0, 'epoch': 127, 'lr': 0.000125, 'n_epochs': 500, 'test_loss': 0.02568249670892126, 'time': 13939.861764431, 'train_loss': 0.019775001585936893, 'val_loss': 0.03675157647757303}
            , {'run': "056___fhn-gru", '_runtime': 4183.6232569, '_step': 15, '_timestamp': 1742490490.8742442, '_wandb': {'runtime': 4306}, 'batch_size': 128, 'best_epoch': 15, 'best_test': 0.06614899209567479, 'best_train': 0.06748723043572336, 'best_val': 0.06988165243750527, 'coeff_train': 0.5852923851898851, 'epoch': 15, 'lr': 0.0005, 'n_epochs': 500, 'test_loss': 0.06614899209567479, 'time': 4142.826177120209, 'train_loss': 0.06748723043572336, 'val_loss': 0.06988165243750527}
            , {'run': "058___fhn-set4", '_runtime': 14987.1812247, '_step': 78, '_timestamp': 1742518522.038277, '_wandb': {'runtime': 14989}, 'batch_size': 256, 'best_epoch': 58, 'best_test': 0.026159123284742236, 'best_train': 0.019671335610512056, 'best_val': 0.03538567855139263, 'coeff_train': 0.4655132331658358, 'epoch': 78, 'lr': 0.00025, 'n_epochs': 200, 'test_loss': 0.02384521224303171, 'time': 14915.380064487455, 'train_loss': 0.01652296915111181, 'val_loss': 0.04077457508537918}
            , {'run': "059___fhn-set4-gru", '_runtime': 12729.269175724, '_step': 122, '_timestamp': 1742561825.093523, '_wandb': {'runtime': 12730}, 'batch_size': 256, 'best_epoch': 102, 'best_test': 0.03794126462889835, 'best_train': 0.005722407972146021, 'best_val': 0.02459011127939448, 'coeff_train': 0.697412129302523, 'epoch': 122, 'lr': 0.000125, 'n_epochs': 200, 'test_loss': 0.03839862856693799, 'time': 12707.918282747269, 'train_loss': 0.004537066876104004, 'val_loss': 0.03128099467721768}
            , {'run': "060___fhn-set4-old", '_runtime': 7848.923654686, '_step': 115, '_timestamp': 1742556992.6555638, '_wandb': {'runtime': 7849}, 'batch_size': 256, 'best_epoch': 95, 'best_test': 0.03578539553564042, 'best_train': 0.012222266184950346, 'best_val': 0.031619106186553836, 'coeff_train': 0, 'epoch': 115, 'lr': 0.00025, 'n_epochs': 200, 'test_loss': 0.02979799290187657, 'time': 7834.196428775787, 'train_loss': 0.008440809986112933, 'val_loss': 0.0350145691918442}
        ]

        sweep_test3 = [
            {'run': "rural-sweep-1--new", '_runtime': 17462.796973622, '_step': 139, '_timestamp': 1742603220.945015, '_wandb': {'runtime': 17464}, 'batch_size': 256, 'best_epoch': 119, 'best_test': 0.030992308777058497, 'best_train': 0.007364878394199829, 'best_val': 0.027395313343731686, 'coeff_train': 0.33505770049087247, 'epoch': 139, 'lr': 0.000125, 'n_epochs': 200, 'test_loss': 0.02920797813567333, 'time': 17438.81091785431, 'train_loss': 0.0059961314264096715, 'val_loss': 0.030780186265474185}
            , {'run': "astral-sweep-2--gru", '_runtime': 19809.434212023, '_step': 185, '_timestamp': 1742623037.4117568, '_wandb': {'runtime': 19810}, 'batch_size': 256, 'best_epoch': 165, 'best_test': 0.037519468256505206, 'best_train': 0.004345168837493188, 'best_val': 0.02628742677916307, 'coeff_train': 0.4821195112136664, 'epoch': 185, 'lr': 0.000125, 'n_epochs': 200, 'test_loss': 0.035830627093673684, 'time': 19788.18929862976, 'train_loss': 0.003011567703766846, 'val_loss': 0.03380165572161786}
            , {'run': "scarlet-sweep-3--old", '_runtime': 9186.905935175, '_step': 160, '_timestamp': 1742632239.425357, '_wandb': {'runtime': 9188}, 'batch_size': 256, 'best_epoch': 140, 'best_test': 0.03074078288045712, 'best_train': 0.005162834616279916, 'best_val': 0.02126801016856916, 'coeff_train': 0, 'epoch': 160, 'lr': 0.000125, 'n_epochs': 200, 'test_loss': 0.034998951421584934, 'time': 9174.278196811676, 'train_loss': 0.00436640358529985, 'val_loss': 0.02286278140672948}
            , {'run': "noble-sweep-4--new", '_runtime': 13913.518979332, '_step': 111, '_timestamp': 1742654193.4651668, '_wandb': {'runtime': 13914}, 'batch_size': 256, 'best_epoch': 91, 'best_test': 0.042153427202720195, 'best_train': 0.015057576359494736, 'best_val': 0.03691689274273813, 'coeff_train': 0.473680185599509, 'epoch': 111, 'lr': 0.000125, 'n_epochs': 200, 'test_loss': 0.03774792823242024, 'time': 13889.762244224548, 'train_loss': 0.013327732245976987, 'val_loss': 0.03917645325418562}
            , {'run': "celestial-sweep-5--gru", '_runtime': 14214.554279027, '_step': 132, '_timestamp': 1742668413.9365375, '_wandb': {'runtime': 14215}, 'batch_size': 256, 'best_epoch': 112, 'best_test': 0.05861798900878057, 'best_train': 0.006000793176262002, 'best_val': 0.03405513160396367, 'coeff_train': 0.46192254292628854, 'epoch': 132, 'lr': 0.000125, 'n_epochs': 200, 'test_loss': 0.058182089822366834, 'time': 14193.034108161926, 'train_loss': 0.0050498503603433305, 'val_loss': 0.036278822721214965}
            , {'run': "proud-sweep-6--old", '_runtime': 6978.469322632, '_step': 120, '_timestamp': 1742675400.450749, '_wandb': {'runtime': 6979}, 'batch_size': 256, 'best_epoch': 100, 'best_test': 0.040670332906302065, 'best_train': 0.009412769755152496, 'best_val': 0.026279589044861495, 'coeff_train': 0, 'epoch': 120, 'lr': 0.000125, 'n_epochs': 200, 'test_loss': 0.03931117242609616, 'time': 6965.9404809474945, 'train_loss': 0.007832211192305151, 'val_loss': 0.02774870308348909}
            , {'run': "radiant-sweep-7--new", '_runtime': 13094.5732265, '_step': 94, '_timestamp': 1742665538.5315635, '_wandb': {'runtime': 13095}, 'batch_size': 256, 'best_epoch': 74, 'best_test': 0.025359623919939622, 'best_train': 0.01516830631482758, 'best_val': 0.03943296280340291, 'coeff_train': 0.5574094081201759, 'epoch': 94, 'lr': 0.00025, 'n_epochs': 200, 'test_loss': 0.02404491585912183, 'time': 13040.379710197449, 'train_loss': 0.012050226021950182, 'val_loss': 0.041520741855492815}
            , {'run': "icy-sweep-8--gru", '_runtime': 11890.6906085, '_step': 95, '_timestamp': 1742677438.2187934, '_wandb': {'runtime': 11892}, 'batch_size': 256, 'best_epoch': 75, 'best_test': 0.03795734868617728, 'best_train': 0.013747355030653509, 'best_val': 0.03767710510874167, 'coeff_train': 0.5285459040804684, 'epoch': 95, 'lr': 0.000125, 'n_epochs': 200, 'test_loss': 0.03706227568909526, 'time': 11865.988898038864, 'train_loss': 0.011840765332607065, 'val_loss': 0.03904054296435788}
            , {'run': "stellar-sweep-9--old", '_runtime': 10048.8503206, '_step': 125, '_timestamp': 1742687493.841606, '_wandb': {'runtime': 10050}, 'batch_size': 256, 'best_epoch': 105, 'best_test': 0.02184576947183814, 'best_train': 0.016133316870974867, 'best_val': 0.03413675882620737, 'coeff_train': 0, 'epoch': 125, 'lr': 6.25e-05, 'n_epochs': 200, 'test_loss': 0.02124250223278068, 'time': 10029.506676912308, 'train_loss': 0.015118980118514676, 'val_loss': 0.03502187333651818}
        ]

        temp = [
            {'run': "", }
        ]


        output = {
            "default_code_same_dim": default_code_same_dim,
            "new_LSTM": new_LSTM,
            "x_update_alpha": x_update_alpha,
            "x_update_alpha_opt": x_update_alpha_opt,
            "x_update_beta": x_update_beta, 
            "improving": improving,
            "sweep": sweep, 
            "hyperparams": hyperparams,
            "sweep_test1": sweep_test1, 
            "sweep_test2": sweep_test2, 
            "vdp_gru": vdp_gru,
            "fhn_model": fhn_model,
            "sweep_test3": sweep_test3, 
            } if all else {
            #"sweep_test2": sweep_test2, 
            #"vdp_gru": vdp_gru,
            #"fhn_model": fhn_model,
            "sweep_test3": sweep_test3, 
            }

        return output


    def get_from_csv(self, loc):
        results = []
        n = 0 if loc=="run_data/csv_files/Finals_fhn.csv" else 0

        with open(loc, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in itertools.islice(reader, n, None):#reader:
                try:
                    summary = ast.literal_eval(row['summary'])
                    name = row.get('name', 'unknown_run')
                    summary['run'] = name
                    results.append(summary)
                except Exception as e:
                    print(f"Errore nella riga {row}: {e}")

        return {'dir_passed': results}

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
            "test_loss": 0.2,
            "train_loss": 0.2,
            "val_loss": 0.4,
            "best_epoch": 0.1,
            "time": 0.1
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
            tradeoff_function__2 = lambda run: (run["val_loss"] * 0.6 + run["test_loss"] * 0.3 + run["train_loss"] * 0.1) * (1 + run["best_epoch"] / 100)

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

            # Find minimum value and its corresponding x_value
            min_y = y_values.min()
            min_x = x_values.loc[y_values.idxmin()]
            min_run = df_sorted.loc[y_values.idxmin(), "run"]

            # Plot each dataset's metric trajectory
            plt.plot(x_values, y_values, marker='o', linestyle='-', label=dataset_name)
            plt.scatter(min_x, min_y, color='red', s=100, zorder=3)
            plt.text(min_x, min_y, f"{min_y:.4f}", fontsize=10, ha='right', va='bottom', color='red')
            plt.text(min_x, min_y, f"{min_run}", fontsize=10, ha='left', va='bottom', color='blue')

            print("Minimum for metric", metric, "in data set", dataset_name, "is", min_y, "for run", min_run)
        print("\n")

        plt.xlabel("Training Steps (or Epochs)")
        plt.ylabel(metric.replace("_", " ").capitalize())  # Formatting metric name for display
        plt.title(f"Comparison of {metric} Across Datasets")
        plt.legend()
        plt.grid(True)
        plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    args = CalcValues.parse_arguments()
    CalcValues(display=args.display, plot=args.plot, all=args.all, loc=args.loc)


"""
Minimum for metric val_loss in data set default_code_same_dim is 0.026048427477242456 for run 026
Minimum for metric val_loss in data set sweep is 0.02143756610651811 for run radiant-sweep-4
Minimum for metric val_loss in data set hyperparams is 0.028925311145564868 for run 047


Minimum for metric test_loss in data set default_code_same_dim is 0.026321270325708957 for run 026
Minimum for metric test_loss in data set sweep is 0.08527710055193258 for run glamorous-sweep-1
Minimum for metric test_loss in data set hyperparams is 0.03224825329842076 for run 047


Minimum for metric train_loss in data set default_code_same_dim is 0.014251226118258223 for run 027
Minimum for metric train_loss in data set sweep is 0.007751289033207786 for run hearty-sweep-2
Minimum for metric train_loss in data set hyperparams is 0.0073168578018094335 for run 047


Minimum for metric time in data set default_code_same_dim is 7328.914924144745 for run 026
Minimum for metric time in data set sweep is 4381.372041940689 for run faithful-sweep-4
Minimum for metric time in data set hyperparams is 6442.120647192001 for run 044
"""