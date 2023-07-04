import csv
import pandas as pd
import sys
import glob
from os import path
import math

if __name__ == "__main__":
    csv_dir_root = sys.argv[1]
    app = csv_dir_root.split('/')[1]
    df = pd.DataFrame()
    csv_dirs = path.join(csv_dir_root, 'W*')
    for csv_file in glob.iglob(csv_dirs + '/*model_step_1_keras_log.csv'):
        dataset_in_df = pd.read_csv(csv_file, sep=';')
        df = pd.concat([df, dataset_in_df.tail(1)], ignore_index=True)
    columns_list = ['N_bytes_DF_output_loss', 'N_bytes_UF_output_loss', 'N_agg_DF_output_loss',
                    'N_agg_UF_output_loss', 'N_agg_DF_output_loss', 'N_agg_UF_output_loss', 'time']
    summary_dict = {'App': app,
                    'ceil_mean_epoch': math.ceil(df['epoch'].mean()),
                    'std_dev_epoch:': df['epoch'].std()}
    for col in columns_list:
        if col in df.columns:
            summary_dict['mean_' + col] = df[col].mean()
            summary_dict['std_dev_' + col] = df[col].std()
        else:
            summary_dict['mean_' + col] = 'N/A'
            summary_dict['std_dev_' + col] = 'N/A'

    summary_csv_filename = 'summary_preliminary_exp.csv'
    mask_header = False if path.isfile(summary_csv_filename) else True
    with open(summary_csv_filename, 'a') as file:
        writer = csv.writer(file)
        if mask_header:
            writer.writerow(summary_dict.keys())
        writer.writerow(summary_dict.values())

