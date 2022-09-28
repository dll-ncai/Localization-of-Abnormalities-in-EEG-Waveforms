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
from tqdm import tqdm
import gc
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = [224/100,224/100]

'''

This file contains functions used to extract windows from EDF files and labels from CSV files.
extractlabels function iterates over CSV files to extract the labels and time stamps of the present abnormalities.
cleanlabels function classifies each label into a super class of labels
generatelabelarray creates a label array of the same dimensions as the EEG data


EDF Files are read and are used to extract the EEG data. 
The data is split in to windows(epochs) of 2 seconds (400 samples) with 1 second overlap
Each window is labeled as abnormal if atleast 25% of abnormality is present in it

'''



############################################################################################################

def extractlabels(name, csvdir):
  if name is not int:
    name = str(name) + '.csv'
  if csvdir[-1] != '/':
    csvdir = csvdir + '/'

  csv = pd.read_csv(csvdir + name)
  offset = csv['File Start'][0].split(':')
  
  beg = csv['Start time'].str.split(':').to_numpy()
  end = csv['End time'].str.split(':').to_numpy()
  #print(beg)
  channels = csv['Channel names'].str.split().to_numpy()
  type = csv['Comment'].to_numpy()
  for i in range(len(offset)):
    offset[i] = int(offset[i])
  for i in range(beg.shape[0]):
    if len(beg[i]) < 3:
      continue
    for j in range(len(beg[i])):
      beg[i][j] = int(beg[i][j])
      if len(beg[i]) < 4:
        beg[i].append(0)
    for j in range(len(end[i])):
      end[i][j] = int(end[i][j])
      if len(end[i]) < 4:
        end[i].append(0)
    for j in range(len(offset)):
      #print(beg[i][j])
      beg[i][j] = beg[i][j] - offset[j]
      end[i][j] = end[i][j] - offset[j]
    beg[i] = int((beg[i][0]*3600 + beg[i][1]*60 + beg[i][2] + beg[i][3]/1000)*200)
    end[i] = int((end[i][0]*3600 + end[i][1]*60 + end[i][2] + end[i][3]/1000)*200)
  #print(beg)
  return type, channels, beg, end

############################################################################################################

def cleanlabels(labels):
  label_dict = [['No Comment', 'delete previous', 'nan', 'Normal'], ['sharp waves', 'sharp wave', 'Sharp Wave'], ['delta slow waves','delta waves', 'delta slow waves', 'sharp and delta slow waves', 'sharp and slow waves', 'sharp and slow wave', 'sharp and slow wave ','slowing wave', 'sharp and slow waves','sharp and slow wave','generalized paroxysmal delta slow waves', 'generalized delta slow waves', 'delta slow wave', 'slow waves', 'generalized delta slow waves ', 'paroxysmal delta slow waves', ' delta slow waves', 'delta waves', 'paroxysmal generalized delta slow waves', 'paroxysmal generalized deta slow waves','Delta ','sharp and delta slow waves', 'Delta Slow Wave'], ['2 hertz slow spike and wave discharge','spike wave','spikes','polypspikes and wave','polyspikes and wave', 'generalized paroxysmal spike and wave discharge', 'fragmented spike and wave discharge', 'generalized paroxysmal 3 hertz spike and wave discharge', 'generalized paroxysmal  spike and wave discharge', 'generalized spike and wave discharge', 'generalized 4 hertz spike and wave discharge', 'spike and wave', 'spike and wave discharge', 'generalized 3 hertz spike and wave discharge', 'generalized 2 hertz spike and wave discharge', 'spike and waves', '3 hertz fragmented spike and wave discharge', 'generalized spike and wave ', 'generalized spike and wave', 'spike and wave ', 'polyspikes discharge', 'Generalized  paroxysmal 4 hertz spike and wave discharge', 'generalized 3.5 hertz spike and wave discharge', 'Generalized 3 hertz spike and wave discharge', 'Generalized 2 hertz spike and wave discharge', '3 hertz spike and wave discharge', ' 3 hertz spike and wave discharge', 'Generalized spike and wave discharge', 'paroxysmal generalized 3.5 spike and wave discharge', 'paroxysmal generalized 3.5 hertz spike and wave discharge', ' spike and wave discharge', 'generalized spike  and wave discharge', 'generalized  3 hertz spike and wave discharge', 'generalized  spike and wave discharge', 'generalized 3 hertz  spike and wave discharge', 'spike an dwave', 'spike', 'Paroxysmal generalized 3 hertz spike and wave discharge', 'paroxysmal generalized spike and wave discharge', 'polyspikes', 'paroxysmal generalized 3 hertz spike and wave discharge', ' generalized spike and wave discharge', 'generalized polyspike discharge', 'generalized polyspikes discharge', 'paroxysmal generalized 4 hertz spike and wave discharge', 'rolandic spike', 'generalized paroxysmal 3.5 spike and wave discharge', 'spike and wave discharge 3 hertz', 'fragemented spike and wave discharge', 'fragmented 3 hertz spike and wave discharge', 'generalized  2 hertz spike and wave discharge', 'generalized 2 hertz  spike and wave discharge', 'parosysmal generalized 3 hertz spike and wave discharge', 'spiek and wave', 'generalized 2.5 hertz spike and wave discharge' ,'sharp wave', 'sharp waves','spike and wave', 'spike and wave ','polyspikes and wave', 'polyspikes ', 'polyspikes', 'spikes', 'polypspikes and wave', 'spike wave', 'spike and wave','polyspikes ','Spike and Wave Discharge'], ['Beta waves', 'beta waves', 'Beta Wave'], ['theta waves', 'Theta Wave'], ['triphasic waves', 'Triphasic Wave'], ['low voltage','no waveform', 'Low Voltage']]
  ret = []
  for i in range(len(labels)):
    for j in range(len(label_dict)):
      for k in range(len(label_dict[j])):
        if label_dict[j][k] == labels[i]:
          labels[i] = label_dict[j][-1]
          ret.append(label_dict[j][-1])
  if len(ret) != len(labels):
    print("Did not catch all labels. Please check")
    return labels
  else:
    return ret

############################################################################################################

def generatelabelarray(labels, channels):
  labels, channels = np.array(labels), np.array(channels)
  ret = []      #Make sure to check channel names in channel list. Python was not working fine, could not confirm channel names, also unprocessed edf does not contain FZ PZ and CZ, check if 7 or 8 additional leads
  #channel_list = ['Fp1-A1','Fp2-A2','F3-A1','F4-A2','C3-A1','C4-A2','P3-A1','P4-A2','O1-A1','O2-A2','F7-A1','F8-A2','T3-A1','T4-A2','T5-A1','T6-A2','Add_lead1','Add_lead2','Add_lead3']
  channel_list = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','PZ','CZ']
  for i in range(labels.shape[0]):
    ret.append([])
    for k in range(len(channel_list)):
      if channel_list[k] in channels[i]:
        ret[-1].append(labels[i])
      else:
        ret[-1].append("Normal")
  return ret

############################################################################################################

def generatewindows(data, labels, beg, end, windowsize = 2, min_ab_threshold = 0.7):
  nsamples = windowsize * 200
  data, labels, beg, end = np.array(data), np.array(labels), np.array(beg), np.array(end)
  windows, windowlabel = [], []
  for i in range(int(data.shape[1]/nsamples)):
    windows.append([])
    windowlabel.append([])
    for j in range(data.shape[0]):
      windows[-1].append(data[j][i*nsamples:(i+1)*nsamples])
      windowlabel[-1].append("Normal")
  print(np.shape(windowlabel))
  #for labels which are larger than file
  windowlabel = np.array(windowlabel)
  new_end = np.delete(end,np.argwhere(end/nsamples >= windowlabel.shape[0]-1))
  print('Warning:Delete lables :',(len(end)-len(new_end)))
  new_beg = beg[0:len(new_end)]
  windowlabel = windowlabel.tolist()
  #broadcasting labels
  for j in range(len(new_beg)):
    window_index_beg = new_beg[j]/nsamples
    if (1 - window_index_beg % 1) > min_ab_threshold:
      window_index_beg = int(window_index_beg)
    else:
      window_index_beg = math.ceil(window_index_beg)
    #print(window_index_beg)
    windowlabel[window_index_beg] = labels[j]
    window_index_end = new_end[j]/nsamples
    if window_index_end % 1 > min_ab_threshold:
      window_index_end = math.ceil(window_index_end)
    else: 
      window_index_end = int(window_index_end)
    #print('length of window label', len(windowlabel[window_index_end]))
    #print('length of    labels[j]',len(labels[j]))
    windowlabel[window_index_end] = labels[j]
    net_window = window_index_end - window_index_beg
    #print('Window Diffrence :',net_window)
    for i in range(int(net_window)):
      windowlabel[window_index_beg + i] = labels[j]
    #print('labels are:',windowlabel[window_index_end])
    #except:
      #print(int(end[j]/nsamples), j)
  return windows, windowlabel

############################################################################################################

def sample_truncator(data,threshold):
  data = data[threshold:]
  data = data[:data.shape[0]-threshold]
  return data

############################################################################################################

def sample_labeling(data,labels,beg,end):           #For assigning each sample a label to later pass on to MNE function for making epochs of labels
  sample_label = []
  labels, beg, end = np.array(labels), np.array(beg), np.array(end)
  for i in range(data.shape[1]):
    sample_label.append([])
    for j in range(data.shape[0]):
      sample_label[-1].append("Normal")
  sample_label = np.array(sample_label)
  print(np.shape(sample_label))
  new_end = np.delete(end,np.argwhere(end >= sample_label.shape[0]))
  print('Warning:Delete lables :',(len(end)-len(new_end)))
  new_beg = beg[0:len(new_end)]
  sample_label = sample_label.tolist()
  for k in range(len(new_beg)):
    sample_label[new_beg[k]] = labels[k]
    sample_label[new_end[k]] = labels[k]
    diff = new_end[k]-new_beg[k]
    for l in range(int(diff)):
      sample_label[new_beg[k]+l] = labels[k]
  return sample_label

############################################################################################################
# Thresholding Windows to assign whole window one label. Threshold is set in percentage.
 
def epoch_windowing(epochs_labels, threshold_val = 25):      #threshold is in percentage
  label_array = []
  threshold = round(epochs_labels.shape[2]*(threshold_val/100))
  for i in range(epochs_labels.shape[1]):
    label_array.append([])
    for j in range(epochs_labels.shape[0]):
      if np.count_nonzero(epochs_labels[j][i]) > threshold:
        ab_type = np.unique(epochs_labels[j][i])[-1]
        label_array[-1].append(ab_type)
      else:
        label_array[-1].append(0)

  print(np.unique(label_array))
  label_array = np.array(label_array, dtype = int)
  label_array = np.transpose(label_array)
  
  print('Shape of Label Epochs:', label_array.shape)   #(no of epochs, channels)
  return label_array

############################################################################################################

def int_encoder(label_data):      #Integer encoding Labels
  y_train = label_data
  y_train[y_train == 'Normal'] = 0
  y_train[y_train == 'Delta Slow Wave'] = 1
  y_train[y_train == 'Sharp Wave'] = 2
  y_train[y_train == 'Spike and Wave Discharge'] = 2 
  y_train  = y_train.astype('int')
  return y_train
############################################################################################################

#For Extracting Features of Wavelet Decompositions
def stat_features(sub_band):      # Max value, Min value, Mean, Variance, Standard Deviation
  max = np.max(sub_band)
  min = np.min(sub_band)
  mean = np.mean(sub_band)
  variance  = np.var(sub_band)
  std = np.std(sub_band)
  return max, min, mean, variance, std


def ram_release():
  gc.collect()
