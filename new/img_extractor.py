import numpy as np
import os
import random
import matplotlib.pyplot as plt
import mne
import pywt
from utils import*
from tqdm import tqdm
import gc 
from random import randint
# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['figure.figsize'] = [224/100,224/100]
import time
import multiprocessing
import pandas as pd
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
csvdir = '/home/dll-1/Desktop/eeg/datasets/Adil paper/csv/SW & SSW CSV Files'     # Directory of CSV Files
edfdir = '/home/dll-1/Desktop/eeg/datasets/Adil paper/edf/Abnormal EDF Files'	  # Directory of Abnormal EDF files
main_path = 'scallogramTest/' 
normalEdf_dir = '/home/dll-1/Desktop/eeg/datasets/Adil paper/edf/Normal EDF Files/'
data_csv = '/home/dll-1/Desktop/eeg/datasets/Adil paper/data_split.csv'



Label_data = np.empty((0,19))
Epochs_data = np.empty((0,19,400))
dest_list = ['Normal', 'Slowing Waves', 'Spike and Sharp waves']  # Names of subfolders with in main folder
# done_folder = 'data/doneFiles/'   #After processing each CSV file, the CSV files are moved to this folder
waveletsTypes = ['mexh','morl', 'gaus1', 'gaus2']
splits = ['train', 'valid', 'test']

allFileList = os.listdir(csvdir)
allEdfList = os.listdir(normalEdf_dir)


df = pd.read_csv(data_csv)


train_abnormal = df[(df['Class'] == 'abnormal') & (df['Type'] == 'train')]
train_abnormal = train_abnormal['File'].tolist()

test_abnormal = df[(df['Class'] == 'abnormal') & (df['Type'] == 'test')]
test_abnormal = test_abnormal['File'].tolist()

valid_abnormal = df[(df['Class'] == 'abnormal') & (df['Type'] == 'valid')]
valid_abnormal = valid_abnormal['File'].tolist()


train_normal = df[(df['Class'] == 'normal') & (df['Type'] == 'train')]
train_normal = train_normal['File'].tolist()

test_normal = df[(df['Class'] == 'normal') & (df['Type'] == 'test')]
test_normal = test_normal['File'].tolist()

valid_normal = df[(df['Class'] == 'normal') & (df['Type'] == 'valid')]
valid_normal = valid_normal['File'].tolist()




def imageExtraction(full_data,scales,waveletType,main_path,split,dest_list,file_num,Label_data,Epochs_data,channelNum):
    print('Working on Channel: ', channelNum)
    print('Epochs Data: ',Epochs_data.shape[0])
    ts = time.time()
    i = channelNum
    window = []
    coef,_ = pywt.cwt(full_data[i], scales , waveletType, method = 'conv')

    for j in range(Epochs_data.shape[0]):
        sig_cwt,_ = pywt.cwt(Epochs_data[j][i], scales , waveletType,method = 'conv')
        if Label_data[j][i] == 1:
            #print(1)
            plt.imshow(sig_cwt, extent=[1, 31, 31, 1], cmap='nipy_spectral', vmax=abs(coef).max(), vmin=-abs(coef).max())
            plt.axis('off')
            plt.savefig(fname = main_path + waveletType  + '/' + split + '/' + dest_list[1] + '/' + 'img_' + str(j) + '_' + str(i) + '_' + str(file_num) + '.png', bbox_inches = 'tight')
            plt.close()

        elif Label_data[j][i] == 2:
            #print(4)
            plt.imshow(sig_cwt, extent=[1, 31, 31, 1], cmap='nipy_spectral', vmax=abs(coef).max(), vmin=-abs(coef).max())
            plt.axis('off')
            plt.savefig(fname = main_path + waveletType  + '/' + split + '/' + dest_list[2] + '/' + 'img_' + str(j) + '_' + str(i) + '_' + str(file_num) + '.png', bbox_inches = 'tight')
            plt.close()
        
        else:
            if j % 30 == 0:
                plt.imshow(sig_cwt, extent=[1, 31, 31, 1], cmap='nipy_spectral', vmax=abs(coef).max(), vmin=-abs(coef).max())
                plt.axis('off')
                plt.savefig(fname = main_path + waveletType  + '/' + split + '/' + dest_list[0] + '/' + 'img_' + str(j) + '_' + str(i) + '_' + str(file_num) + '.png', bbox_inches = 'tight')
                plt.close()
    te = time.time()
    print('Tota Time Taken for Channel:',channelNum, te-ts)


for waveletType in waveletsTypes:
    if not os.path.exists(main_path + waveletType):
        os.makedirs(main_path + waveletType)

    for split in splits:
        if not os.path.exists(main_path + waveletType + '/' + split + '/' + dest_list[0]):
            os.makedirs(main_path + waveletType + '/' + split + '/' + dest_list[0])
            os.makedirs(main_path + waveletType + '/' + split + '/' + dest_list[1])
            os.makedirs(main_path + waveletType + '/' + split + '/' + dest_list[2])
    
    for split in splits:

        if split == 'train':
            split_files = train_abnormal

        elif split == 'test':
            split_files = test_abnormal

        else:
            split_files = valid_abnormal

        print('Split Type: ', split)
        print('Split Files :', split_files)
        
        for file in split_files:

            print('#######  IMPORTING EDF FILES #######')
            file_num = int(file[:-4])
            print(file_num)
            edf_name = str(10000000 + file_num)[1:] + '.edf' 
            raw = mne.io.read_raw_edf(edfdir + "/" + edf_name,preload = True,exclude = ['A1','A2'])     # Importing all EEG Channels, exculding A1 A2 since matlab has already refrenced the channels with A1 and A2
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


            num_processes = Epochs_data.shape[1]

            # You can provide a list of tuples as the second argument to map, where each tuple contains the input arguments
            inputs = [(full_data,scales,waveletType,main_path,split,dest_list,file_num,Label_data,Epochs_data,i) for i in range(num_processes)]

            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.starmap(imageExtraction, inputs)


            collected = gc.collect()
            print('Gc collect',collected)

        collected = gc.collect()
        print('Gc collect',collected)


    ################################## NORMALL ###############################
    
    for split in splits:


        if split == 'train':
            split_files = train_normal

        elif split == 'test':
            split_files = test_normal

        else:
            split_files = valid_normal

        print('Going now Normal with wavelet type', waveletType)
        random.seed(444)     #Make sure the seed is same to get similar results
        window_num = 150     #Define the number of windows you want per normal file
        '''
        This is done to downsample from the large amount of normal data present. This is done to mitigate the data imbalance
        You can get the total number of normal windows(images) produced = window_num * no of files. 
        '''

        win_ch = int(window_num/19)
        coef_data = np.empty((2,19))
        for file in tqdm(split_files):
            raw = mne.io.read_raw_edf(normalEdf_dir + file,preload = True,exclude = ['A1','A2'])     # Importing all EEG Channels, exculding A1 A2 since matlab has already refrenced the channels with A1 and A2
            raw.filter(l_freq=0.5,h_freq=45,fir_window='hamming')      # Bandpass filtering [1-45] Hz
            full_data = raw.get_data()
            epochs=mne.make_fixed_length_epochs(raw,duration=2,overlap=0)
            epochs_data=epochs.get_data()  

            print('Shape of input data after Epochs:',epochs_data.shape)

            for i in range(18):
                coef,_ = pywt.cwt(full_data[i], scales,waveletType,method = 'conv')
                for j in range(win_ch):
                    rand_num = randint(0,epochs_data.shape[0]-1)
                    sig_cwt,_ = pywt.cwt(epochs_data[rand_num][i], scales , 'mexh',method = 'conv')
                    plt.imshow(sig_cwt, extent=[1, 31, 31, 1], cmap='nipy_spectral', vmax=abs(coef).max(), vmin=-abs(coef).max())
                    plt.axis('off')
                    plt.savefig(fname = main_path + waveletType  + '/' + split + '/' + dest_list[0] + '/' + 'img_' + str(rand_num) + '_' + str(i) + '_' + str(file[:-4]) + '.png', bbox_inches = 'tight')
                    plt.close()
            collected = gc.collect()
            print('Gc collect',collected)
        

