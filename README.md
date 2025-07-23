# EEG-Based Cognitive Workload Classification using CNN 

This project explores cognitive workload classification using EEG signals and Convolutional Neural Networks (CNNs). It utilizes real EEG data recorded during mental arithmetic tasks, sourced from the PhysioNet repository. The objective is to distinguish between **high** and **low** mental workload based on brain activity.

---
## Dataset

- **Name**: EEG During Mental Arithmetic Tasks
- **Source**: [PhysioNet](https://physionet.org/content/eegmat/1.0.0/)
- **Subjects**: 36 individuals (24 Good performance, 12 Poor performance)
- **Recordings**: Two per subject —  
  - `_1.edf`: Background EEG  
  - `_2.edf`: EEG during arithmetic task  
- **Duration**: 60 seconds per recording
- **Channels**: 23-channel EEG system (standard 10/20 montage)

> See full dataset description and contributor info [here](https://physionet.org/content/eegmat/1.0.0/).

---


