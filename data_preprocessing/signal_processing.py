# %%
import pandas as pd
import multiprocessing as mp
import numpy as np
import os
import glob

from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

pd.set_option("display.max_columns", None)


def process_data(save_path:str, csv_file_list:list) -> None:
    """Processing Data

    Args:
        save_path (str): Save Path for ecg file
        csv_file_list (list): raw ecg file list
    """
    for file_nm in csv_file_list:
        df_ecg = pd.read_csv(file_nm, header=None, sep="\n", engine='python')
        df_ecg = df_ecg.assign(parsed = lambda x: x[0].apply(lambda x: (x.split(','))), 
                            indicator = lambda x: x['parsed'].apply(lambda y: len(y)))
        
        save_processed_file_path = csv_file_list[0][:-20] + "Processed"
        
        if os.path.exists(save_processed_file_path):
            pass
        else:
            os.makedirs(save_processed_file_path)
        
        df_ecg.query('indicator == 4')[0].reset_index(drop=True).to_csv(os.path.join(save_processed_file_path, csv_file_list[0][-11:]), header=False, index=False)
        
        df_ecg = pd.read_csv(os.path.join(save_processed_file_path, csv_file_list[0][-11:]), delimiter=',|"', header=None, engine='python')
        df_ecg = df_ecg.loc[:, 1:4]
        df_ecg.columns = ['times', 'ecg', 'time_point', 'segment_id']
        
        for segment_id in list(set(df_ecg['segment_id'])):
            np.save(file=os.path.join(save_path, segment_id)+'.npy', arr=df_ecg.query("segment_id == @segment_id")['ecg'].values) 
            
    return None
# %%
def main():
    ECG_DATAPATH = "../data/KEMDy19/ECG"
    
    if os.path.exists(os.path.join(ECG_DATAPATH, 'processed')):
        SAVEPATH = os.path.join(ECG_DATAPATH, 'processed')
        pass
    else:
        os.makedirs(os.path.join(ECG_DATAPATH, 'processed'))
        SAVEPATH = os.path.join(ECG_DATAPATH, 'processed')
    
    csv_file_list = glob.glob(os.path.join(ECG_DATAPATH, "*/Original/*"))
    
    file_list = np.array_split(np.array(csv_file_list), mp.cpu_count())
    
    process_map(partial(process_data, SAVEPATH), file_list, max_workers=mp.cpu_count())
# %%
if __name__ == '__main__':
    main()