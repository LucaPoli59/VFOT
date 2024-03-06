"""
File che serve trovare il risultato migliore
"""

import os
import json

from main_dir import constant as const

squared_error_score = {}
absolute_error_score = {}

date_dirs = os.listdir(const.SAVE_PATH)
for date_dir in date_dirs:
    time_dirs = os.listdir(const.SAVE_PATH + date_dir)
    for time_dir in time_dirs:
        final_path = const.SAVE_PATH + date_dir + "\\" + time_dir

        config_file = open(final_path + "\\config.json")
        metrics = json.load(config_file)['model']['metrics']['Test_original']
        config_file.close()
        squared_error, absolute_error = metrics["mean_squared_error"]['LSTM'], metrics["mean_absolute_error"]['LSTM']

        model_name = date_dir + "_" + time_dir

        squared_error_score[model_name] = float(squared_error)
        absolute_error_score[model_name] = float(absolute_error)


best_squared_error_score = sorted(squared_error_score.items(), key=lambda x: x[1])
best_absolute_error_score = sorted(absolute_error_score.items(), key=lambda x: x[1])


print("Best squared error:\n")
for score in best_squared_error_score:
    print("File: ", score[0] + "\t\tValue: ", score[1])

print("\n\n\nBest absolute error:\n")
for score in best_absolute_error_score:
    print("File: ", score[0] + "\t\tValue: ", score[1])
