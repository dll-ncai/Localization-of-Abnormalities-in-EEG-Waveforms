import numpy as np
from PIL import Image
import itertools
import os
import shutil
import random
import glob
import math
import matplotlib.pyplot as plt
import pandas as pd
import mne
import pywt
import utils
from utils import*
from tqdm import tqdm
import gc
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = [224/100,224/100]

'''

This file iterates over the CSV files to extract the labels and time stamps of the present abnormalities.
EDF Files are read and are used to extract the EEG data. 
The data is split in to windows(epochs) of 2 seconds (400 samples) with 1 second overlap
Each window is labeled as abnormal if atleast 25% of abnormality is present in it
CWT is taken for each window 
Each img is saved in the respective folders with the format 'img_window#_channel#_file#.png' ( To keep track of each image produced )
After extracing the images out of each file the CSV File is moved to the 'done folder'
This code is memory managed, but it can still take upto 7-8 hours to produce images of around 113 CSV files

'''

scales = np.arange(1,24)        # Scales for CWT
csvdir = '/home/FYP/Desktop/FINAL DATASET ~ 31st Aug/CSV Files/SW & SSW CSV Files'     # Directory of CSV Files
edfdir = '/home/FYP/Desktop/FINAL DATASET ~ 31st Aug/EDF Files/Abnormal EDF Files/'	  # Directory of Abnormal EDF files
Label_data = np.empty((0,19))
Epochs_data = np.empty((0,19,400))
main_path = '/home/FYP/Desktop/new_imgs_test/'       #Path of main folder which contains the subfolder of each abnormality
dest_list = ['Slowing Waves', 'Normal', 'Spike and Sharp waves']  # Names of subfolders with in main folder
done_folder = '/home/FYP/Desktop/FINAL DATASET ~ 31st Aug/CSV Files/temp/'   #After processing each CSV file, the CSV files is moved to this folder


for file in os.listdir(csvdir):
    
    print('#######  IMPORTING EDF FILES #######')
    file_num = int(file[:-4])
    print(file_num)
    edf_name = str(10000000 + file_num)[1:] + '.edf' 
    raw = mne.io.read_raw_edf(edfdir+edf_name,preload = True,exclude = ['A1','A2'])     # Importing all EEG Channels, exculding A1 A2 since matlab has already refrenced the channels with A1 and A2
    raw.filter(l_freq=1,h_freq=45)      # Bandpass filtering [1-45] Hz
    full_data = np.array(raw.get_data())
    epochs=mne.make_fixed_length_epochs(raw,duration=2,overlap=1)           #Setting overlapping duration of 1 second
    epochs_data=epochs.get_data()
    print('#######  CATCHING CSV FILES #######')
    type,channels,beg,end = extractlabels(file_num,csvdir)
    type = cleanlabels(type)
    type = cleanlabels(type)
    type, channels, beg, end = np.array(type), np.array(channels), np.array(beg), np.array(end)
    labels = np.array(generatelabelarray(type, channels))
    #print(labels)
    data = np.array(raw.get_data())
    label_data = np.transpose(sample_labeling(data,labels,beg,end))
    print('Shape of label_data before epochs:',np.shape(label_data))
    label_data = int_encoder(label_data)
    info_labels = mne.create_info(ch_names=['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'PZ', 'CZ'],sfreq = 200)   #Setting channels in MNE Object
    raw_labels = mne.io.RawArray(label_data,info_labels)                #Making MNE Raw object and making overlapping epochs of labels per sample
    epochs_labels=mne.make_fixed_length_epochs(raw_labels,duration=2,overlap=1)         #Setting overlapping duration of 1 second
    epochs_labels = epochs_labels.get_data()
    epochs_labels = epoch_windowing(epochs_labels) #(no of epochs, channels)
    Label_data = np.array(Label_data, dtype = int)

    Label_data = epochs_labels
    Epochs_data = epochs_data
    
    print('Shape of epochs_of_sample_labels:',epochs_labels.shape)      #(no of epochs, channels, samples)
    print(np.unique(Label_data,return_counts = True))
    print(Label_data.shape)
    print(Epochs_data.shape)
    print(full_data.shape)
    for i in tqdm(range(Epochs_data.shape[1])):
        window = []
        coef,_ = pywt.cwt(full_data[i], scales , 'mexh',method = 'conv')
        if len(np.unique(Label_data)) == 2:
            if np.unique(Label_data)[-1] == 1:
                ab_type = 1
                window = np.argwhere(Label_data.T[i] == ab_type)
                for win_no in window:
                    win_no = np.squeeze(win_no)
                    sig_cwt,_ = pywt.cwt(Epochs_data[win_no][i], scales , 'mexh',method = 'conv')           # Setting the mother wavelet of CWT and scaled of (1,24)
                    plt.imshow(sig_cwt, extent=[-1, 1, 31, 1], cmap='nipy_spectral', aspect='auto',vmax=abs(coef).max(), vmin=-abs(coef).max())         #image color normalization
                    plt.axis('off')
                    plt.savefig(fname = main_path + dest_list[0] + '/' + 'img_' + str(win_no) + '_' + str(i) + '_' + str(file_num) + '.png', bbox_inches = 'tight')
                    plt.close()
                    
            else:
                ab_type = 2
                window = np.argwhere(Label_data.T[i] == ab_type)
                for win_no in window:
                    win_no = np.squeeze(win_no)
                    sig_cwt,_ = pywt.cwt(Epochs_data[win_no][i], scales , 'mexh',method = 'conv')
                    plt.imshow(sig_cwt, extent=[-1, 1, 31, 1], cmap='nipy_spectral', aspect='auto',vmax=abs(coef).max(), vmin=-abs(coef).max())
                    plt.axis('off')
                    plt.savefig(fname = main_path + dest_list[2] + '/' + 'img_' + str(win_no) + '_' + str(i) + '_' + str(file_num) + '.png', bbox_inches = 'tight')
                    plt.close()
        else:
           for j in range(Epochs_data.shape[0]):
                sig_cwt,_ = pywt.cwt(Epochs_data[j][i], scales , 'mexh',method = 'conv')
                if Label_data[j][i] == 1:
                    #print(1)
                    plt.imshow(sig_cwt, extent=[-1, 1, 31, 1], cmap='nipy_spectral', aspect='auto',vmax=abs(coef).max(), vmin=-abs(coef).max())
                    plt.axis('off')
                    plt.savefig(fname = main_path + dest_list[0] + '/' + 'img_' + str(j) + '_' + str(i) + '_' + str(file_num) + '.png', bbox_inches = 'tight')
                    plt.close()
                elif Label_data[j][i] == 2:
                    #print(4)
                    plt.imshow(sig_cwt, extent=[-1, 1, 31, 1], cmap='nipy_spectral', aspect='auto',vmax=abs(coef).max(), vmin=-abs(coef).max())
                    plt.axis('off')
                    plt.savefig(fname = main_path + dest_list[2] + '/' + 'img_' + str(j) + '_' + str(i) + '_' + str(file_num) +'.png', bbox_inches = 'tight')
                    plt.close()

    shutil.move(csvdir + '/'+str(file_num) + '.csv',done_folder +str(file_num)+'.csv')
    print('Moving CSV file to done folder')
    gc.collect()