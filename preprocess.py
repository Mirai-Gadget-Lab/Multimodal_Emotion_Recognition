# %%
import pandas as pd
from glob import glob
import os
import shutil
import matplotlib.pyplot as plt
import argparse

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--root_path', required=True, type=str)
    p.add_argument('--save_path', default='./data/', type=str)
    config = p.parse_args()

    return config   

def check_path_exist(x, data_path):
    wav_name = x+'.wav'
    txt_name = x+'.txt'
    wav = os.path.exists(os.path.join(data_path, wav_name))
    txt = os.path.exists(os.path.join(data_path, txt_name))
    if wav & txt:
        return True
    else:
        return False
    
def main(args):
    
    # segment_id로 충분히 핸들링 할수 있으므로, 굳이 여러 디렉토리에 나눠져있을 필요가없음
    # wav, txt를 하나의 디렉토리에 저장. 
    try:
        os.mkdir(args.save_path)
    except:
        pass 
    
    try: 
        os.mkdir(os.path.join(args.save_path, 'txt_wav'))
    except:
        pass
    
    datas = glob(os.path.join(args.root_path, 'wav/*/*/*'))
    for path in datas:
        try:
            shutil.copy(path, os.path.join(args.save_path, 'txt_wav/%s'%os.path.basename(path)))
        except IsADirectoryError:
            pass
        
    annotations = sorted(glob(os.path.join(args.root_path, 'annotation/*.csv')))
    
    # 모든 Annotation 읽은이후 하나의 DataFrame으로 변환
    df_ls = []
    for path in annotations:
        df = pd.read_csv(path)
        df_ls.append(df.loc[1:, "Wav":"Unnamed: 12"])
        
    csv = pd.concat(df_ls)

    # Emoition Label이 2개이상인 데이터 제거 (즉 헷갈리는 데이터는 제거)
    # 컬럼을 정리한 이후, annotation file 저장
    
    csv = csv[csv['Total Evaluation'].map(lambda x: ';' not in x)]
    
    csv.columns = ['wav_start', 'wav_end', 'ecg_start',
                'ecg_end', 'eda_start', 'eda_end', 
                'temp_start', 'temp_end', 'segment_id',
                'emotion', 'valence', 'arousal']
    csv = csv[['segment_id', 'emotion', 'valence', 'arousal',
        'wav_start', 'wav_end', 'ecg_start',
        'ecg_end', 'eda_start', 'eda_end', 
        'temp_start', 'temp_end']]
    
    # 생체신호를 안쓰기때문에, 중복된 segment_id를 제거
    
    csv = csv.drop_duplicates(subset=['segment_id'], ignore_index=True)
    
    # 파일이 누락된 경우 제거
    csv['path_exist'] = csv['segment_id'].map(lambda x: check_path_exist(x, os.path.join(args.save_path, 'txt_wav/')))
    csv[csv['path_exist']].to_csv(os.path.join(args.save_path, 'annotation.csv'), index=False)

if __name__ == '__main__':
    args = define_argparser()
    main(args)