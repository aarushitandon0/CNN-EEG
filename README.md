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

## Study Background

> **Original Study Title**: Electroencephalograms during Mental Arithmetic Task Performance  
> **Authors**: Zyma I, Tukaev S, Seleznov I, Kiyono K, Popov A, Chernykh M, Shpenkov O  
> **Published in**: *Data* (MDPI), 2019, 4(1):14  
> **DOI**: [10.3390/data4010014](https://doi.org/10.3390/data4010014)

Subjects performed serial subtraction tasks (e.g., 3141 − 42). EEGs were recorded using the **Neurocom EEG 23-channel system**. All signals were artifact-free and filtered. ICA was used to remove eye, muscle, and cardiac noise.

> EEGs were recorded monopolarly, electrodes placed according to the international 10-20 scheme, referenced to ear electrodes.

---

## Project Objective

To classify EEG recordings into **low** and **high** cognitive workload categories using a deep learning model trained on preprocessed EEG epochs.

---

##  Tech Stack

### **Languages**
-  Python 3

### **Signal Processing & EEG Tools**
- [MNE](https://mne.tools/stable/index.html) — EEG signal reading, filtering, visualization from `.edf` files

### **Machine Learning**
- [TensorFlow](https://www.tensorflow.org/) + Keras — CNN model for binary classification
- [scikit-learn](https://scikit-learn.org/) — data preprocessing, normalization, train-test split

### **Data Handling**
- NumPy — matrix and array operations  
- Pandas — loading and manipulating CSV label data

### **Visualization**
- Matplotlib — EEG waveform plots, PSD plots, accuracy/loss graphs

### **Development Environment**
- Jupyter Notebook or Google Colab (recommended)
- VSCode (for modular `.py` development)

### **Dataset**
-  [UCI EEG Arithmetic Dataset](https://archive.ics.uci.edu/ml/datasets/eeg+database)

### **Version Control**
- Git & GitHub — code sharing, documentation, collaboration

## Project Flow

This section outlines the complete pipeline for classifying EEG-based cognitive workload using deep learning.

---

### Step 1: Dataset & Labels

- Downloaded EEG data from **[PhysioNet EEG-MAT](https://physionet.org/content/eegmat/1.0.0/)**.
- Each subject has 2 files:
  - `*_1.edf` → Baseline EEG
  - `*_2.edf` → EEG during task
- Labels are defined as:
  - `0`: High workload (bad performance)
  - `1`: Low workload (good performance)
- Label data is stored in `subject-info.csv`.

---

### Step 2: Load & Preprocess EDF Files (MNE)

- Loaded `.edf` files using `mne.io.read_raw_edf()`.
- Renamed and filtered EEG channels.
- Set standard 10-20 EEG montage for channel location mapping.
- Sample preprocessing:
  ```python
  raw = mne.io.read_raw_edf(filename, preload=True)
  raw.rename_channels(lambda x: x.replace("EEG ", ""))
  raw.set_montage('standard_1020')
  raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)

### Step 3: Visualization (EEG Analysis)
Used MNE’s built-in plotting to understand data quality and features:
- raw.plot() — raw signal inspection
- raw.plot_psd(fmax=50) — Power Spectral Density
- raw.plot_sensors() — EEG channel layout on scalp

### Step 4: Epoch Creation
- EEG data was segmented into 2-second overlapping windows using:
```
epochs = mne.make_fixed_length_epochs(raw_filtered, duration=2.0, overlap=1.0)
```
- These short epochs act as individual samples for classification.
- Each epoch of shape: (19 channels × 128 samples)
- Dataset shape:
- ```
  X shape: (67665, 19, 128)
  y shape: (67665,)
   ```

### Step 5: Feature Extraction (Frequency Bands)
Computed Power Spectral Density (PSD) using Welch's method:
```
psds, freqs = raw.compute_psd(fmin=0.5, fmax=45).get_data(return_freqs=True)
```
#### Band Power Feature Table

> **Effective Window Size**: `4.096 s`  
> **Note**: `pick_types()` is deprecated. Use `inst.pick(...)` instead.

| Channel | Delta       | Theta       | Alpha       | Beta        | Gamma       |
|---------|-------------|-------------|-------------|-------------|-------------|
| Fp1     | 2.43e-11    | 4.31e-12    | 1.35e-12    | 7.17e-13    | 2.46e-13    |
| Fp2     | 8.94e-12    | 5.47e-12    | 1.35e-12    | 6.61e-13    | 2.22e-13    |
| F3      | 9.63e-12    | 4.83e-12    | 1.58e-12    | 6.11e-13    | 1.81e-13    |
| F4      | 1.31e-11    | 7.03e-12    | 1.98e-12    | 7.36e-13    | 1.86e-13    |
| F7      | 9.96e-12    | 4.80e-12    | 2.52e-12    | 7.93e-13    | 2.98e-13    |
| F8      | 1.39e-11    | 2.99e-12    | 1.22e-12    | 5.76e-13    | 1.62e-13    |
| T3      | 1.53e-11    | 3.05e-12    | 1.25e-12    | 4.98e-13    | 1.88e-13    |
| T4      | 1.58e-11    | 2.75e-12    | 1.24e-12    | 5.89e-13    | 1.49e-13    |
| C3      | 1.06e-11    | 3.80e-12    | 1.45e-12    | 4.92e-13    | 1.73e-13    |
| C4      | 1.12e-11    | 4.49e-12    | 1.53e-12    | 5.55e-13    | 1.66e-13    |
| T5      | 1.15e-11    | 2.88e-12    | 1.22e-12    | 5.94e-13    | 1.93e-13    |
| T6      | 1.54e-11    | 2.81e-12    | 1.26e-12    | 6.60e-13    | 1.56e-13    |
| P3      | 1.13e-11    | 3.45e-12    | 1.47e-12    | 5.71e-13    | 1.70e-13    |
| P4      | 1.25e-11    | 3.21e-12    | 1.58e-12    | 5.64e-13    | 1.65e-13    |
| O1      | 1.36e-11    | 3.71e-12    | 2.04e-12    | 7.56e-13    | 1.88e-13    |
| O2      | 1.45e-11    | 3.75e-12    | 2.08e-12    | 7.46e-13    | 1.91e-13    |
| Fz      | 1.62e-11    | 9.65e-12    | 2.49e-12    | 7.50e-13    | 1.94e-13    |
| Cz      | 1.34e-11    | 5.85e-12    | 1.79e-12    | 5.76e-13    | 1.88e-13    |
| Pz      | 1.47e-11    | 3.61e-12    | 1.56e-12    | 5.81e-13    | 1.72e-13    |
| ECG ECG | 5.55e-11    | 1.51e-11    | 5.15e-12    | 2.14e-12    | 4.40e-13    |

---

 **EEG Channels Used**:  
`['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz', 'ECG ECG']`

 **Band Power Rows Extracted**: `5`

Extracted average power in frequency bands for each channel:

- Delta (1–4 Hz)
- Theta (4–8 Hz)
- Alpha (8–12 Hz)
- Beta (12–30 Hz)
- Gamma (30–40 Hz)


### Step 6: Train-Test Split
EEG samples and labels were reshaped and split:

```
X = all_epochs.reshape(-1, 19, 128, 1)  # shape for CNN
y = all_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Step 7: CNN Model for Classification
Convolutional Neural Network designed with TensorFlow/Keras.

Architecture:

- 2 Conv2D + MaxPooling layers
- Flatten + Dense + Dropout
- Output: Sigmoid activation for binary classification

Compiled & trained:
```
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

### Test Accuracy
- Test Accuracy: 82.6%
- Final model shows strong ability to classify cognitive workload.


## EEG Power Spectral Density (PSD) Plot
![Image](https://github.com/aarushitandon0/CNN-EEG/blob/main/images/Screenshot%202025-07-24%20022105.png)

The Power Spectral Density (PSD) visualization is generated using:

```python
epochs.plot_psd(fmin=1, fmax=40)
```

> *Note*: `plot_psd()` is a legacy MNE function. For newer implementations, you should use `.compute_psd().plot()`.

This plot shows the **distribution of power** (in dB/Hz) over the **frequency range 1–40 Hz** across all EEG channels.

####  What it tells us:
- **X-axis**: Frequency (Hz)
- **Y-axis**: Power (dB/Hz re 1 µV²)
- Each colored line represents a different EEG channel.
- The curves represent how power varies over frequencies — typically:
  - **Delta (1–4 Hz)**: High during deep sleep.
  - **Theta (4–8 Hz)**: Light sleep, meditation.
  - **Alpha (8–12 Hz)**: Relaxed state, closed eyes.
  - **Beta (12–30 Hz)**: Active thinking, alertness.
  - **Gamma (>30 Hz)**: Complex processing, perception.

#### Interpretation:
- Peaks in different bands may correlate with **cognitive workload**, mental state, or neural activity patterns.
- This visual is key for **feature extraction** and **band power analysis** for downstream classification (like CNN-based workload prediction).

---

## EEG Topomap Visualization

The EEG signals were recorded using the standard **10–20 International System**, covering 19 scalp locations including frontal (F), temporal (T), central (C), parietal (P), and occipital (O) regions.

### Electrode Layout

![Topomap Layout](https://github.com/aarushitandon0/CNN-EEG/blob/main/images/Screenshot%202025-07-24%20022355.png)

This layout illustrates the electrode positions used in the experiment. Key electrodes like **Fz, Cz, Pz** are crucial for detecting cognitive workload changes.

### Channel Importance

- **Frontal Region (F3, F4, Fz):** Associated with attention and working memory
- **Central Region (C3, Cz, C4):** Useful in motor and cognitive control
- **Parietal/Occipital (Pz, O1, O2):** Visual processing and cognitive load
- **Temporal (T3, T4, T5, T6):** Involved in language and auditory tasks

This topographical map helps in understanding the **spatial distribution of brain activity**, and is used both during preprocessing and in interpreting CNN attention.

## Raw EEG Signal Plot
![image3](https://github.com/aarushitandon0/CNN-EEG/blob/main/images/Screenshot%202025-07-24%20022655.png)

Before proceeding with any preprocessing or modeling, it's essential to **visually inspect the raw EEG signals** to identify artifacts, baseline shifts, or abnormalities.

We used MNE's `raw.plot()` function for this purpose:

```python
raw.plot(n_channels=20, scalings='auto', duration=10, start=0)
```
- n_channels=20: Displays 20 EEG channels.
- scalings='auto': Auto scales the amplitude based on the signal range.
- duration=10: Shows 10 seconds of signal per screen.
- start=0: Begins at the start of the recording.

#### Sample Raw EEG Output
As seen above, each channel represents the potential difference over time, giving insight into brain activity. Channels like Fp1, Cz, Pz, O1 are used in analysis, and ECG is often removed to reduce noise.
This visual inspection is a key step before bandpass filtering, artifact removal, or applying machine learning models.

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

