# NeuroAssist: Open-Source Automatic Event Detection in Scalp EEG

* [Introduction](#introduction)
* [Requirements](#requirements)
* [Dataset](#dataset)
* [Citation](#citation)

This repository contains the source code for 'NeuroAssist: Open-Source Automatic Event Detection in Scalp EEG' using data from **"NUST-MH-TUKL EEG"**. The study compares the performance of state-of-the-art machine learning and deep learning algorithms on the task of EEG abnormality localization on the **"NMT-Events"** dataset. The details of our research on localizing abnormalities in EEG data can be found in the following paper:

* Muhammad Ali Alqarni, Hira Masood, Adil Jowad Qureshi, Muiz Alvi, Haziq Arbab, Hassan Aqeel Khan, Awais Mehmood Kamboh, Saima Shafait, and Faisal Shafait (2024) * **"NeuroAssist: Open-Source Automatic Event Detection in Scalp EEG",** IEEE Access. doi: (10.1109/ACCESS.2024.3492673)

# Requirements
1. Install mne library available at https://github.com/mne-tools/mne-python
2. This code was programmed in Python 3.10 (might work for other versions also)
3. Dataset available at: https://dll.seecs.nust.edu.pk/downloads/

# Dataset
The NMT-Events consists of labeled EEG data from the years 2018 to 2021. EEG waveforms are present in European Data Format (EDF) files. These are 19 channel EEG waveforms of patients from the Military Hospital (MH) sampled at a frequency of 200 Hz. Timestamps along with labels for the abnormalities are stored in Comma Seperated Value (CSV) files. There are a total of 962 Normal and 113 Abnormal EEG recordings.

# Citation
This repo was used to generate the results for the following paper on Pre-Diagnostic Screening of Abnormal EEG.
  
[//]: # (  > Citation: Alqarni MA, Masood H, Qureshi AJ, Alvi M, Arbab H, Khan HA, Kamboh AM, Shafait S, and Shafait F &#40;2024&#41; **NeuroAssist: Open-Source Automatic Event Detection in Scalp EEG** IEEE Access &#40;__&#41;)
  > Citation: M. Ali Alqarni et al., "NeuroAssist: Open-Source Automatic Event Detection in Scalp EEG," in IEEE Access, vol. 12, pp. 170321-170334, 2024, doi: 10.1109/ACCESS.2024.3492673. 
**BibTex Reference:**
