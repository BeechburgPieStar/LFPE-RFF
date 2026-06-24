# LFPE-RFF: Label-Free Performance Estimation for RF Fingerprinting

# Submit to ICCC

This repository provides the implementation of **Label-Free Performance Estimation for RF Fingerprinting Under Varying Channels**. The goal is to estimate the target-domain identification accuracy of a deployed RF fingerprinting model without using target-domain labels.

The method builds an energy-accuracy mapping from augmented source data and then predicts the accuracy of unlabeled target data using only model outputs.

## Overview

Radio frequency fingerprinting (RFF) identifies wireless devices by learning hardware-specific signal characteristics from received I/Q samples. However, real-world channel changes, receiver differences, and environmental variations can cause distribution shifts, which may significantly degrade the accuracy of a trained classifier after deployment.

This project addresses the following question:

**Can we estimate the target-domain accuracy of an RFF classifier without access to target labels?**

The proposed framework uses energy-based indicators extracted from classifier logits. During the calibration stage, source data are augmented to simulate distribution shifts. For each augmented dataset, the framework computes:

* Mean energy from model logits
* Ground-truth accuracy using retained source labels

A regression model is then fitted to map mean energy to accuracy. During inference, the target-domain mean energy is computed and passed to the regression model to estimate the target accuracy.

## Main Features

* Label-free target-domain performance estimation
* Energy-based performance indicator
* Augmentation-based distribution shift construction
* Linear regression from energy to accuracy
* Support for RF fingerprinting models such as MSCAN, CVCNN, and ResNet
* Experiments on the WiSig ManyRx dataset

## Method Pipeline

The framework consists of two stages.

### Stage 1: Calibration Dataset Generation

The source data are transformed using signal-level augmentations that emulate channel and hardware variations. For each augmented dataset, the trained classifier produces logits, and the average energy score is computed.

The output is a CSV file containing energy-accuracy pairs.

### Stage 2: Predictor Training and Target Accuracy Estimation

A regression model is trained using the generated calibration CSV. The trained predictor maps average energy to classification accuracy.

For the unlabeled target domain, the classifier logits are used to compute the average energy. The predictor then estimates the target-domain accuracy.

## Repository Structure

```text
.
├── step1_generate_dataset.py        # Generate calibration energy-accuracy dataset
├── step2_train_and_predict.py       # Train predictor and estimate target accuracy
├── datasets/
│   └── load_dataset.py              # Dataset loading utilities
├── models/
│   ├── get_model.py                 # Model construction
│   ├── tools.py                     # Utility functions
│   └── augmentation.py              # Signal augmentation functions
├── weights/                         # Pretrained classifier weights
├── results/                         # Generated CSV files and predictors
└── README.md
```

## Requirements

The code requires Python and the following packages:

```bash
pip install numpy pandas scipy scikit-learn joblib torch
```

Depending on your environment, please install the appropriate PyTorch version from the official PyTorch website.

## Dataset Preparation

The experiments use the **WiSig ManyRx** dataset. By default, the scripts expect the dataset root directory to be:

```bash
/datasets
```

The default dataset name is:

```bash
WiSig_ManyRx
```

The scripts use receiver:

```bash
rx_1-1
```

and use four day-based splits:

```text
data1, data2, data3, data4
```

A leave-one-day-out protocol is used. For example, when `data1` is selected as the test split, the remaining splits `data2`, `data3`, and `data4` are used as the training domain.

## Pretrained Weights

Before running the estimation pipeline, prepare the pretrained classifier weights under the following structure:

```text
weights/
├── data123/
│   └── MSCAN_WiSig_ManyRx_best.pt
├── data124/
│   └── MSCAN_WiSig_ManyRx_best.pt
├── data134/
│   └── MSCAN_WiSig_ManyRx_best.pt
└── data234/
    └── MSCAN_WiSig_ManyRx_best.pt
```

The folder name indicates the training splits. For example:

```text
weights/data234/MSCAN_WiSig_ManyRx_best.pt
```

is used when `data1` is the test split.

You may replace `MSCAN` with other supported models, such as `CVCNN` or `ResNet`, if the corresponding model definitions and weights are available.

## Usage

### Step 1: Generate the Calibration Dataset

Run the following command to generate the calibration CSV for a specific model and test split:

```bash
python step1_generate_dataset.py \
    --base_dir /datasets \
    --dataset WiSig_ManyRx \
    --wisig_test_split data1 \
    --rx 1-1 \
    --samples_per_tx 200 \
    --model_name MSCAN \
    --device cuda:0 \
    --output_dir results
```

This step will:

1. Load the training-domain data
2. Apply predefined signal augmentations
3. Compute model logits
4. Compute energy scores
5. Evaluate accuracy on augmented source data
6. Save the energy-accuracy pairs to a CSV file

The generated file will be saved as:

```text
results/MSCAN_data1.csv
```

The CSV contains columns such as:

```text
Config, AWGN_sev, Phase_sev, PA_sev, Total_sev, Accuracy, Avg_Energy
```

### Step 2: Train the Predictor and Estimate Target Accuracy

After generating the calibration CSV, run:

```bash
python step2_train_and_predict.py \
    --base_dir /datasets \
    --dataset WiSig_ManyRx \
    --wisig_test_split data1 \
    --rx 1-1 \
    --samples_per_tx 200 \
    --model_name MSCAN \
    --degree 1 \
    --device cuda:0 \
    --output_dir results
```

This step will:

1. Load the calibration CSV
2. Train a regression predictor from average energy to accuracy
3. Load the target test split
4. Compute the target-domain average energy
5. Predict the target-domain accuracy
6. Save the predictor and prediction result

The outputs are:

```text
results/MSCAN_data1_predictor.joblib
results/MSCAN_data1_result.csv
```

## Running Multiple Test Splits

To evaluate all four day-based test splits, run:

```bash
for split in data1 data2 data3 data4
do
    python step1_generate_dataset.py \
        --base_dir /datasets \
        --dataset WiSig_ManyRx \
        --wisig_test_split $split \
        --rx 1-1 \
        --samples_per_tx 200 \
        --model_name MSCAN \
        --device cuda:0 \
        --output_dir results

    python step2_train_and_predict.py \
        --base_dir /datasets \
        --dataset WiSig_ManyRx \
        --wisig_test_split $split \
        --rx 1-1 \
        --samples_per_tx 200 \
        --model_name MSCAN \
        --degree 1 \
        --device cuda:0 \
        --output_dir results
done
```

## Important Arguments

| Argument             | Description                        | Default            |
| -------------------- | ---------------------------------- | ------------------ |
| `--base_dir`         | Dataset root directory             | `/datasets`        |
| `--dataset`          | Dataset name                       | `WiSig_ManyRx`     |
| `--wisig_test_split` | Target test split                  | `data1` or `data2` |
| `--rx`               | Receiver index                     | `1-1`              |
| `--samples_per_tx`   | Number of samples per transmitter  | `200`              |
| `--model_name`       | Classifier model name              | `MSCAN`            |
| `--degree`           | Polynomial degree for regression   | `1`                |
| `--energy_T`         | Temperature for energy computation | `1.0`              |
| `--batch_size`       | Batch size                         | `128`              |
| `--device`           | Computation device                 | `cuda:0`           |
| `--output_dir`       | Directory for results              | `results`          |

## Output Files

### Calibration CSV

Generated by `step1_generate_dataset.py`.

Example:

```text
results/MSCAN_data1.csv
```

Main fields:

* `Config`: augmentation configuration name
* `AWGN_sev`: AWGN severity level
* `Phase_sev`: phase perturbation severity level
* `PA_sev`: nonlinear distortion severity level
* `Total_sev`: total augmentation severity
* `Accuracy`: classification accuracy on augmented source data
* `Avg_Energy`: average energy score

### Predictor File

Generated by `step2_train_and_predict.py`.

Example:

```text
results/MSCAN_data1_predictor.joblib
```

This file stores the trained regression predictor and metadata.

### Prediction Result CSV

Generated by `step2_train_and_predict.py`.

Example:

```text
results/MSCAN_data1_result.csv
```

Main fields:

* `Model`
* `Test_Split`
* `Avg_Energy`
* `True_Accuracy`
* `Predicted_Accuracy`
* `Error`

## Example Output

A typical prediction output looks like:

```text
Prediction Result
============================================================
  Avg Energy:         -8.1234
  True Accuracy:      88.95%
  Predicted Accuracy: 96.53%
  Error:              7.58%
============================================================
```

## Notes

* The method assumes that the augmentation-generated distribution shift space sufficiently covers the target-domain shift.
* If the target-domain average energy is outside the calibration energy range, the script will print a warning.
* The target labels are only used for evaluation and error reporting in experiments. The estimated accuracy itself is obtained from the energy-based predictor.
* For real deployment, target labels are not required.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{xu2025lfpe_rff,
  title     = {Label-Free Performance Estimation for RF Fingerprinting Under Varying Channels},
  author    = {Xu, Qichen and Da, Xinyang and Wang, Yu},
  booktitle = {To appear},
  year      = {2026}
}
```
