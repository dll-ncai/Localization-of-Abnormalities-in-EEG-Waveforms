# Localization-of-Abnormalities-in-EEG-Waveforms

* [Introduction](#introduction)
* [Requirements](#requirements)
* [Dataset](#dataset)
* [Citation](#citation)

This repository contains the source code for 'NeuroAssist: An Open-Source Tool for Automatic Event Detection in Scalp EEG' using data form the **"NMT Scalp EEG Dataset v2.0"** of the year 2020. The study compares the performance of state-of-the-art machine learning and deep learning algorithms on the task of EEG abnormality localization on the NMT v2.0 dataset. The details of our research on localizing abnormalities in EEG data can be found in the following paper:

* Muhammad Ali Alqarni, Adil Jowad Qureshi, Muiz Alvi, Haziq Arbab, Hira Masood, Hassan Aqeel Khan, Awais Mehmood Kamboh, Saima Shafait, and Faisal Shafait (2022) * **"NeuroAssist: An Open-Source Tool for Automatic Event Detection in Scalp EEG",** Frontiers in Neuroscience. doi: (insert doi url)

# Requirements
1. Install mne library available at https://github.com/mne-tools/mne-python
2. This code was programmed in Python 3.10 (might work for other versions also)
3. Dataset available at: https://dll.seecs.nust.edu.pk/downloads/

# Dataset
The NMT dataset v2.0 consists of labeled EEG data from the years 2018 to 2021. EEG waveforms are present in European Data Format (EDF) files. These are 19 channel EEG waveforms of patients from the Military Hospital (MH) sampled at a frequency of 200 Hz. Timestamps along with labels for the abnormalities are stored in Comma Seperated Value (CSV) files. There are a total of 962 Normal and 113 Abnormal EEG recordings.

# Citation
This repo was used to generate the results for the following paper on Pre-Diagnostic Screening of Abnormal EEG.
  
  > Citation: Alqarni MA, Qureshi AJ, Alvi M, Arbab H, Masood H, Khan HA, Kamboh AM, Shafait S, and Shafait F (2022) **NeuroAssist: An Open-Source Tool for Automatic Event Detection in Scalp EEG.** Front. Neurosci. (insert doi)

**BibTex Reference:**
