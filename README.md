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



---

## Project Objective

To classify EEG recordings into **low** and **high** cognitive workload categories using a deep learning model trained on preprocessed EEG epochs.

---

## Study Background

> **Original Study Title**: Electroencephalograms during Mental Arithmetic Task Performance  
> **Authors**: Zyma I, Tukaev S, Seleznov I, Kiyono K, Popov A, Chernykh M, Shpenkov O  
> **Published in**: *Data* (MDPI), 2019, 4(1):14  
> **DOI**: [10.3390/data4010014](https://doi.org/10.3390/data4010014)

Subjects performed serial subtraction tasks (e.g., 3141 − 42). EEGs were recorded using the **Neurocom EEG 23-channel system**. All signals were artifact-free and filtered. ICA was used to remove eye, muscle, and cardiac noise.

> EEGs were recorded monopolarly, electrodes placed according to the international 10-20 scheme, referenced to ear electrodes.

---
### Citation & Credit


**Primary Reference**:  
Zyma I, Tukaev S, Seleznov I, Kiyono K, Popov A, Chernykh M, Shpenkov O.  
*Electroencephalograms during Mental Arithmetic Task Performance*.  
Data. 2019; 4(1):14. [https://doi.org/10.3390/data4010014](https://doi.org/10.3390/data4010014)

**PhysioNet Citation**:  
Goldberger AL, Amaral LAN, Glass L, et al.  
*PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.*  
Circulation. 2000;101(23):e215–e220. [RRID:SCR_007345](https://scicrunch.org/resolver/SCR_007345)

---

