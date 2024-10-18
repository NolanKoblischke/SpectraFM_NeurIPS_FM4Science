## Overview

The `model_core/` directory contains the core implementation of **SpectraFM**, our Transformer-based foundation model for analyzing stellar spectra. This model is designed to handle spectral data from any instrument and wavelength range, providing a versatile tool for astrophysical research. The code in this directory is adapted and extended from the foundational work of [Leung & Bovy (2023)](https://arxiv.org/abs/2308.10944), referred to as LB23, and their [Github repository](https://github.com/henrysky/astroNN_stars_foundation). To see examples of the code in use, see `training/` and `results/`.

## Contents

- `stellarperceptron/`: A package containing the main model code.
  - `model.py`: Defines the `StellarPerceptron` class, which implements the Transformer-based architecture for stellar spectroscopy.
  - `layers.py`: Contains custom neural network layers and Transformer blocks tailored for processing spectral data.
  - `nn_utils.py`: Utility functions for neural network operations, including custom loss functions and training utilities.
  - `model_core.py`: Core functions and classes that support the model's configuration, initialization, and training procedures.

## Relation to LB23

The code in `model_core/` is adapted from LB23, which introduced a proof-of-concept Transformer-based foundation model for stellar parameter estimation using Gaia spectra and auxiliary data. Our adaptation extends this model to handle high-resolution stellar spectra from any wavelength range and instrument, enabling the analysis of a broader range of spectroscopic data.

Key adaptations and extensions include:

- Wavelength Encoding Mechanism: We introduced a wavelength positional encoding scheme inspired by the positional encodings in language models. This allows the model to accept spectra with varying wavelength ranges and resolutions, differentiating spectral pixels based on their wavelengths.

- Scaled Context Length: We scaled up the model to accept inputs significantly larger than the original context length used in LB23 (64). This paper uses a context length of 512. We have successfully trained on the full spectra (7514) and are currently analyzing the results which will be detailed in future work.

## Detailed Descriptions

### `stellarperceptron/model.py`

Defines the `StellarPerceptron` class, which serves as the main interface for the Transformer-based model. Key features include:

- **Initialization**: Sets up the model architecture, including the embedding layer, encoder, and decoder, with configurable parameters such as embedding dimensions, number of heads, and dropout rates. See examples for how to use these in `training/`.

- **Model Training** (`fit` method): Implements the training loop, handling data loading, batching, loss computation, and optimization steps. It supports fine-tuning on new datasets and includes options for mixed-precision training and distributed computing. See examples for how to use these in `training/`.

- **Perception and Request Mechanisms**: Implements methods (`perceive` and `request`) for encoding input data and generating predictions. This aligns with the encoder-decoder structure of Transformers, where `perceive` processes the input context, and `request` generates outputs based on specific queries. See examples for how to use these in `results/`.

- **Loading and Saving**: Includes methods to save and load model checkpoints, ensuring that trained models can be reused and shared. See examples for how to use these in `training/`.

### `stellarperceptron/layers.py`

Contains custom neural network layers and modules used in the model:

- **NonLinearEmbedding**: A custom embedding layer that maps input tokens (spectral pixels) to embedding vectors. It includes a non-linear transformation and integrates the wavelength positional encoding.

- **TransformerBlock**: Defines the basic Transformer block used in both the encoder and decoder. It includes multi-head attention, layer normalization, and feed-forward neural network components.

- **StellarPerceptronEncoder and StellarPerceptronDecoder**: Builds upon `TransformerBlock` to create the encoder and decoder parts of the model, respectively. They handle the flow of data through multiple Transformer blocks and manage attention mechanisms.

- **StellarPerceptronTorchModel**: Wraps the embedding layer, encoder, and decoder into one model.

### `stellarperceptron/nn_utils.py`

Provides utility functions and classes for neural network operations:

- **Loss Functions**: Implements the `robust_mean_squared_error` loss function which can handle NaNs and learns to predict an uncertainty estimate. This represents any uncertainty in the training set that cannot be explained by the known variance.

- **TrainingGenerator**: A custom data generator class that handles batching, shuffling, and preparing data for training.

### `stellarperceptron/model_core.py`

Contains core classes and methods that support the overall functionality of the model:

- **StellarPerceptronCore**: An abstract base class that defines common interfaces and utilities for model implementations. It handles tokenization, standardization, and configuration management.

- **Configuration and Standardization**: Provides methods to save and load model configurations, manage input standardization parameters stored in `config.json`.

## Acknowledgements

This code is adapted from the work of LB23.