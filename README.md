# Video Super Resolution

A TensorFlow implementation of a recurrent video super-resolution model developed as an ITI graduation project.

The project focuses on reconstructing high-resolution video frames from low-resolution input sequences using a recurrent neural-network architecture with residual blocks, temporal frame propagation, Gaussian downsampling, and perceptual/image-quality loss functions.

## Project Overview

Video Super Resolution (VSR) aims to improve the spatial resolution of video frames while preserving temporal consistency across consecutive frames. Unlike single-image super-resolution, VSR can use information from neighboring frames to recover details that may be missing or blurred in an individual frame.

This repository implements a learning-based VSR pipeline that:

- Loads high-resolution video frame sequences.
- Generates low-resolution inputs using Gaussian downsampling.
- Trains a recurrent residual model to reconstruct high-resolution frames.
- Uses temporal information from consecutive frames.
- Supports checkpoint-based training and evaluation.
- Includes image-quality and perceptual loss functions.

## Repository Structure

```text
.
├── Architecture.py           # Residual blocks and hidden network architecture
├── model.py                  # Recurrent video super-resolution model and training logic
├── main.py                   # Training configuration and entry point
├── Load_Data.py              # TensorFlow data generator for train/validation sequences
├── Gaussian_DownSample.py    # Gaussian downsampling and data augmentation utilities
├── check_point/              # Model checkpoint directory
└── .gitignore
```

## Main Components

### 1. Model Architecture

The model is built around a recurrent residual network. For each time step, the network consumes consecutive low-resolution frames together with hidden state and previous prediction information.

Key architectural ideas:

- Residual convolutional blocks without batch normalization.
- Hidden-state propagation across frames.
- Previous high-resolution prediction feedback.
- Pixel shuffle / depth-to-space reconstruction for upscaling.
- Pixel unshuffle / space-to-depth handling for recurrent prediction feedback.

### 2. Data Pipeline

The data pipeline loads high-resolution frame sequences and creates low-resolution training inputs dynamically.

The preprocessing flow includes:

1. Load a sequence of high-resolution frames.
2. Apply random data augmentation.
3. Generate low-resolution frames using Gaussian downsampling.
4. Normalize frame values.
5. Batch the data using `tf.data.Dataset`.

### 3. Training Flow

The training loop is implemented inside the model class and performs:

- Sequence-wise recurrent inference.
- Loss calculation between predicted and ground-truth frames.
- Gradient computation using `tf.GradientTape`.
- Optimizer update.
- Periodic checkpoint saving.
- Training and validation loss logging.
- Visual comparison between ground truth and model prediction.

### 4. Loss Functions

The project includes multiple loss options:

- SSIM-based loss.
- PSNR-based loss.
- VGG19 perceptual loss.
- ResNet50 perceptual loss.

The default training configuration in `main.py` uses VGG19-based perceptual loss with AdamW optimization.

## Technical Stack

- Python
- TensorFlow / Keras
- TensorFlow Addons
- NumPy
- Pandas
- Pillow
- SciPy
- Matplotlib

## Dataset Assumption

The training script expects a video-frame dataset arranged as numbered sequence folders. The paths in `main.py` are currently local Windows paths and should be updated before running the project on another machine.

Example expected structure:

```text
dataset/
├── train/
│   ├── 00001/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── ...
└── validation/
    ├── 00001/
    │   ├── 0.png
    │   ├── 1.png
    │   └── ...
    └── ...
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AbdelrahmanElaghoury/Video-Super-Resolution.git
cd Video-Super-Resolution
```

### 2. Install dependencies

Create a virtual environment, then install the required packages:

```bash
pip install tensorflow tensorflow-addons numpy pandas pillow scipy matplotlib
```

### 3. Update dataset paths

Edit the following variables in `main.py`:

```python
GT_train_path = "path/to/train/"
GT_validation_path = "path/to/validation/"
checkpoint_path = "checkpoints/VGG/model_weights"
```

### 4. Start training

```bash
python main.py
```

## Configuration

The main training parameters are defined in `main.py`:

```python
epochs = 50
batch_size = 4
weight_decay = 5e-4
scale_factor = 4
n_f = 128
n_b = 5
```

Where:

- `scale_factor`: super-resolution upscaling factor.
- `n_f`: number of feature channels.
- `n_b`: number of residual blocks.
- `batch_size`: number of sequences per batch.
- `epochs`: number of training epochs.

## What I Learned

This project helped me gain practical experience in:

- Deep learning model development with TensorFlow.
- Video super-resolution concepts.
- Recurrent model design for temporal data.
- Residual CNN architectures.
- Image degradation and Gaussian downsampling.
- Training loops with `tf.GradientTape`.
- Perceptual loss using pretrained CNNs.
- Dataset preprocessing and augmentation.
- Model checkpointing and validation.

## Future Improvements

Potential improvements include:

- Add a `requirements.txt` file.
- Replace hardcoded local paths with command-line arguments.
- Add inference scripts for complete video folders.
- Add quantitative evaluation metrics such as PSNR and SSIM reporting.
- Add sample input/output images.
- Add pretrained weights if available.
- Refactor code formatting for readability and maintainability.
- Add support for modern VSR architectures and benchmark datasets.

## Author

**Abdelrahman Elaghoury**

This repository was developed as part of an ITI graduation project focused on applying deep learning to video super-resolution.
