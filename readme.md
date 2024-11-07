## SpectraFM: Tuning into Stellar Foundation Models

[Paper on OpenReview](https://openreview.net/forum?id=HLEQrER65D)

This repository contains code, data preparation scripts, and model files for SpectraFM, a Transformer-based foundation model designed for cross-instrument spectroscopic analysis in stellar astrophysics. This version was published for the NeurIPS 2024 Foundation Models for Science workshop. Here we focus on APOGEE spectra. SpectraFM leverages synthetic and real spectroscopic data, along with a wavelength encoding mechanism, to predict stellar properties and chemical abundances across multiple datasets and wavelength ranges. Our model can achieve high accuracy in chemical abundance prediction via transfer knowledge from synthetic to real observational data with minimal fine-tuning.

Intermediate training data files and model files will soon be available in Zenodo (waiting for a doi from ArXiv before making a Zenodo upload).

#### `dataset/`

This folder contains scripts and data files used for preparing the dataset. It includes code for loading APOGEE spectra data, removing bad spectra as determined by a series of cuts, cleaning the spectra, and splitting it into training and testing sets suitable for model training and evaluation.

- `create_dataset.py`: Script to load raw APOGEE spectral data, perform cleaning (e.g., removing bad spectra, handling missing values), and split the data into training and testing sets.
- `apogee_wavelength_sol.csv`: Wavelength solution file for the APOGEE spectra, providing the wavelength corresponding to each pixel in the spectra.
- `spectra_all_zeroes_flag.npy`: Used to flag spectra that contain all zeroes and should be excluded from analysis.


#### `model_core/`

Contains the core implementation of the transformer neural network SpectraFM, including the main architecture and utility functions like loading the data for training.

- `stellarperceptron/`:
  - `model.py`: Defines the `StellarPerceptron` class, which implements the Transformer-based neural network architecture used in the paper. Treats stellar spectra as sequences suitable for Transformer models.
  - `layers.py`: Custom neural network layers used in the model, such as attention mechanisms tailored for spectra.
  - `nn_utils.py`: Utility functions for neural network operations, including loss functions and optimization utilities.
  - `model_core.py`: Core functions and classes that support the model, including configurations and model initialization.
- `utils/`: Utility scripts for data handling and plotting.
  - `data_utils.py`: Functions for data manipulation and preprocessing, such as normalization and batching.

#### `training/`

Includes scripts, configurations, and logs related to every step of our training process.

- `1_basemodel_synthetic/`: Training scripts and outputs for the base model pre-trained on synthetic spectra.
  - `trainingAPOGEEGPU.py`: Script used to train the base model on synthetic spectra from ASPCAP covering the full wavelength range (1.515 μm - 1.694 μm) with a context length of 512 pixels.
  - `model_torch/`: Contains the final trained checkpoint `epoch_325`.
- `2_realfinetune_firsthalf/`: Fine-tuning the base model on real spectra from the first half of the wavelength range.
  - `finetuningAPOGEE.py`: Script to fine-tune the pre-trained model on real spectra in the wavelength range of 1.515 μm - 1.603 μm. This step adapts the model to real observational data and is compared to AstroNN models trained on the same data, as discussed in the paper.
  - `epoch_10/`: Directory containing the model checkpoint after fine-tuning for 10 epochs, representing the model used in the subsequent evaluations.
- `3_fefinetune_chunk/`: Scripts for further fine-tuning the model specifically for iron abundance ([Fe/H]) prediction in a particular wavelength chunk (1.611 μm - 1.622 μm) around two prominent iron lines with only 100 iron-rich spectra ([Fe/H] > -1.0).
  - `epoch_120/`: Contains the model checkpoint after 120 epochs of fine-tuning, which is used for the [Fe/H] prediction tasks.
- `basic_NN/`: Implementation of the basic linear neural network trained from scratch on real spectra, serving as a baseline for comparison for the [Fe/H] prediction task (1.611 μm - 1.622 μm) with the same 100 iron-rich spectra.
  - `TrainingNN.py`:
  - `NN_pred_true_may28_4_987_138.csv`: Predictions made by the basic neural network, used for comparison in the results section.
  - `checkpoints/`: Contains the saved model checkpoints for the basic neural network.
- Additional scripts (`*.sh`, `*.out`): Shell scripts and output logs from training sessions.

#### `results/`

Contains scripts and data for evaluating the models and generating the figures presented in the paper.

- `real_spectra_finetune/`: Evaluation scripts and figures related to the model fine-tuned on real spectra (Figure 2).
  - `evaluate_1_basemodel_synthetic_epoch325.py`: Evaluates the base model (pre-trained on synthetic spectra) on the test dataset of real spectra, the initial performance before fine-tuning.
  - `evaluate_2_realfinetune_firsthalf_epoch_10.py`: Evaluates the fine-tuned model on real spectra.
  - `Figure2.py`, `Figure2.png`: shows the performance of the fine-tuned model across key stellar parameters, as presented in the paper.
  - `compare_to_AstroNN.ipynb`: Comparing our model's performance with AstroNN and making Table 1.
- `fe/`: Contains evaluation scripts and figures for the iron abundance prediction task.
  - `EvaluateFeFT.py`: Script to evaluate the model's [Fe/H] predictions using the models.
  - `Figure3.py`, `Figure3.png`: figure comparing [Fe/H] prediction performance across different models, including the basic neural network.
- `attention/`: Figure showing how the model interprets spectra features with attention weights.
  - `extract_attention.py`: Extracts attention weights from the model.
  - `Figure4.py`, `Figure4.png`:Figure showing the attention maps overlaid on spectra.
  - `notable_spectral_lines.csv`: Spectral lines of interest that lined up with attention peaks. From ASPCAP fitting windows.

#### Dependencies

- **Python 3.x**
- `PyTorch`
- See `enviornment.yml` for a complete list of dependencies.
- **Other Dependencies:**
  - Access to APOGEE spectra data files for data preparation. (See Zenodo link)
  - A GPU.

#### Notes

- **Data Availability:** Due to size constraints and licensing, the actual spectral data files (e.g., APOGEE spectra) are not included in the repository. However, we provide intermediate training data files and model files in Zenodo.

- **Configuration:** Adjust file paths and configuration settings in the scripts as needed to match your environment. For example, update paths to the spectral data in `create_dataset.py` and ensure that output directories exist.


#### Acknowledgements

This work builds off of the foundational work of [Leung & Bovy (2023)](https://arxiv.org/abs/2308.10944) and their [repository](https://github.com/henrysky/astroNN_stars_foundation).
