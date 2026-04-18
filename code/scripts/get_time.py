import glob
import pandas as pd
from datetime import datetime

def parse_time(path, name):
    r_times = {}
    for f in glob.glob(path, recursive=True):
        df = pd.read_csv(f)
        if len(df) > 1:
            try:
                # get start and end time from timestamp column
                start_str = str(df['timestamp'].iloc[0])[:26].replace('Z','')
                end_str = str(df['timestamp'].iloc[-1])[:26].replace('Z','')
                start = datetime.fromisoformat(start_str)
                end = datetime.fromisoformat(end_str)
                dur = (end - start).total_seconds()
                
                run = f.split('/')[-3]
                r_times[run] = r_times.get(run, 0) + dur
            except Exception as e:
                print(e)
                pass
    if r_times:
        avg = sum(r_times.values())/len(r_times)
        num_runs = len(r_times)
        print(f"{name}: Total {num_runs} runs. Average Run Time (5 folds): {avg/3600:.2f} hours atau {avg/60:.2f} menit atau {avg:.2f} detik")
        print("Detail per run:")
        for r, ds in r_times.items():
            print(f"  - {r}: {ds/3600:.2f} hours ({ds/60:.2f} menit)")
    else: 
        print(f"{name}: Tidak ada data")

parse_time('code/freeze_encoder/outputs/metrics/**/training_metrics.csv', 'Freeze Encoder')
parse_time('code/lora/outputs/metrics/**/training_metrics.csv', 'LoRA')
