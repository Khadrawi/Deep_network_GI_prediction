import pandas as pd
import papermill as pm
import glob, sys, os, os.path
import shutil
from datetime import datetime, timedelta
from pytz import timezone
from glob import glob
from rich import print
import re
import glob

# set parameters
##########################

animal_lst = ['58-21', '60-21', '87-21', '103-21']
col_names = ["time", "g1", "g2", "g3", "g4", "du1", "du2"]
chan_to_keep = ["time", "g1", "g2", "g3", "g4"]
new_sample_freq = 200
original_sample_freq = 2000
filter_type = 'bandpass' # 'butter_high_only', 'butter_high_low', 'bandpass'
lo_cutoff = 0.05
hi_cutoff = 0.7
use_dropped_packet_removal = True
use_artifact_removal = True
pre_post_pts = 10 # number of pts to remove pre and post for the artifact
fill_frontend = True
    
# remove single files
filelist = glob.glob(os.path.join("*.csv"))
for f in filelist:
    os.remove(f)
filelist = glob.glob(os.path.join("summary.ipynb"))
for f in filelist:
    os.remove(f)

# remove and make directories
if os.path.exists('final_filt_data'):
    shutil.rmtree('final_filt_data')
if os.path.exists('Results'):
    shutil.rmtree('Results')
if not os.path.exists('Results'):
    os.mkdir('Results')

for x in animal_lst:
    print(x)

    results_filename = 'Results/' + x + '_output.ipynb'
    
    pm.execute_notebook('1_templates/analysis.ipynb', results_filename,
                         parameters = dict(animal=x,
                                           col_names=col_names,
                                           chan_to_keep=chan_to_keep,
                                           new_sample_freq=new_sample_freq,
                                           original_sample_freq=original_sample_freq,
                                           filter_type=filter_type,
                                           lo_cutoff=lo_cutoff,
                                           hi_cutoff=hi_cutoff,
                                           use_dropped_packet_removal=use_dropped_packet_removal,
                                           use_artifact_removal=use_artifact_removal,
                                           pre_post_pts=pre_post_pts,
                                           fill_frontend=fill_frontend
                                           ))
    print('------------------------------')
    
    
# Run summary notebook
#####################################

pm.execute_notebook('1_templates/summary.ipynb', 'summary.ipynb')

