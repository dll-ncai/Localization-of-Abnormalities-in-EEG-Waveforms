import numpy as np
from PIL import Image
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import pandas as pd
import mne
import pywt
from tqdm import tqdmf
import gc

# generate random integer values
from random import seed
from random import randint
#%matplotlib inline
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = [224/100,224/100]

'''

EDF Files are read and are used to extract the EEG data. 
The data is split in to windows(epochs) of 2 seconds (400 samples) with 1 second overlap
CWT is taken for each window 
Each img is saved in the respective folders with the format 'img_window#_channel#_file#.png' ( To keep track of each image produced )
This code is memory managed, but it can still take upto 7-8 hours to produce images of around 113 CSV files

'''

scales = np.arange(1,24)        # CWT SCALE
all_files_dir = '/home/FYP/Desktop/FINAL DATASET ~ 31st Aug/EDF Files/Normal EDF Files/'     #Path to all normal EDF files
main_path = '/home/FYP/Desktop/new_imgs_test/normal' # Path to save all the CWT Normal images

random.seed(444)     #Make sure the seed is same to get similar results
window_num = 451     #Define the number of windows you want per normal file
'''
This is done to downsample from the large amount of normal data present. This is done to mitigate the data imbalance
You can get the total number of normal windows(images) produced = window_num * no of files. 
'''

win_ch = int(window_num/19)
coef_data = np.empty((2,19))
for file in tqdm(os.listdir(all_files_dir)):
    raw = mne.io.read_raw_edf(all_files_dir + file,preload = True,exclude = ['A1','A2'])     # Importing all EEG Channels, exculding A1 A2 since matlab has already refrenced the channels with A1 and A2
    raw.filter(l_freq=0.5,h_freq=45,fir_window='hamming')      # Bandpass filtering [1-45] Hz
    full_data = raw.get_data()
    epochs=mne.make_fixed_length_epochs(raw,duration=2,overlap=0)
    epochs_data=epochs.get_data()  

    print('Shape of input data after Epochs:',epochs_data.shape)

    for i in range(18):
        coef,_ = pywt.cwt(full_data[i], scales,'mexh',method = 'conv')
        for j in range(win_ch):
            rand_num = randint(0,epochs_data.shape[0]-1)
            sig_cwt,_ = pywt.cwt(epochs_data[rand_num][i], scales , 'mexh',method = 'conv')
            plt.imshow(sig_cwt, extent=[-1, 1, 31, 1], cmap='nipy_spectral', aspect='auto',vmax=abs(coef).max(), vmin=-abs(coef).max())
            plt.axis('off')
            plt.savefig(fname = main_path + '/' + 'img_' + str(rand_num) + '_' + str(i) + '_' + str(file[:-4]) + '.png', bbox_inches = 'tight')
            plt.close()
    collected = gc.collect()
    print('Gc collect',collected)
