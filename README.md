# Brain MRI Segmentation using Dense Networks

This repository contains PyTorch implementation of dense network-based models for brain MRI segmentation tasks, including hemisphere segmentation and complete brain structure segmentation.

## Overview

The project implements two main segmentation tasks:
- **Hemisphere Segmentation**: Segments brain hemispheres from MRI scans
- **Complete Brain Segmentation**: Performs multi-class segmentation of brain structures

The models are based on DenseNet architecture with custom encoder-decoder designs optimized for medical image segmentation.

## Features

- Dense network architecture with skip connections
- Support for both hemisphere and multi-class brain segmentation
- DICOM to NIfTI conversion pipeline
- Automatic orientation standardization
- Memory-efficient inference on 3D volumes
- Pretrained model support

## Requirements

Install the required dependencies:
```
pip install -r requirements.txt
```

## Model Architecture

The repository includes two main model variants:

### 1. Dense Model
- Used for hemisphere segmentation (2 classes)
- Encoder-decoder architecture with dense blocks
- Skip connections between encoder and decoder

### 2. DenseTV Model
- Used for complete brain segmentation (3+ classes)
- Enhanced decoder with additional feature fusion
- Total variation regularization support

## Usage

### Hemisphere Segmentation

```
python inference_hemi.py
--path_model CODE/model_hemi_mask.pth
--dir_data CODE/Data
--dir_data_dst CODE/Data
--num_classes 2
```

### Complete Brain Segmentation
```
python inference_complete.py
--path_model CODE/model_DenseTV.pth
--dir_data_src CODE/Data
--dir_data_dst CODE/Data
--num_classes 3
```

## Input Data Format

The pipeline expects:
- **DICOM files**: Organized in subject/visit directory structure
- **NIfTI files**: Pre-converted medical images
- **Directory structure**:
```
Data/
├── Subject1/
│ ├── Visit1/
│ │ ├── *.dcm (DICOM files)
│ │ └── *.nii.gz (NIfTI files)
│ └── Visit2/...
├── Subject2/
│ ├── Visit1/
│ │ ├── *.dcm (DICOM files)
│ │ └── *.nii.gz (NIfTI files)
│ └── Visit2/...
*
````

## Output

The inference scripts generate:
- `hemi_mask.nii.gz`: Hemisphere segmentation masks
- `*_seg.nii.gz`: Complete brain segmentation masks
- Standardized NIfTI files with canonical orientation

## Model Components

### Core Classes

- **`_DenseLayer`**: Basic dense layer with batch normalization and dropout
- **`_DenseBlock`**: Multiple dense layers with growth connections
- **`Encoder`**: Feature extraction using dense blocks with downsampling
- **`Decoder`/`DecoderTV`**: Upsampling decoder with skip connections
- **`Dense`/`DenseTV`**: Complete segmentation models

### Key Features

- **Memory Efficiency**: Optional memory-efficient dense layer computation
- **Skip Connections**: Multi-scale feature fusion
- **Pretrained Backbone**: Uses DenseNet-121 pretrained features
- **Flexible Architecture**: Configurable block depths and growth rates

## Configuration Parameters

### Model Parameters
- `block_config`: Tuple defining dense block configurations
- `growth_rate`: Feature map growth rate in dense blocks (default: 32)
- `num_classes`: Number of segmentation classes
- `pretrain`: Use pretrained DenseNet backbone

### Inference Parameters
- `--path_model`: Path to trained model weights
- `--dir_data`: Input data directory
- `--dir_data_dst`: Output directory for results

## Image Preprocessing

The pipeline includes:
1. **Orientation Standardization**: Convert to canonical orientation
2. **Intensity Normalization**: Percentile-based normalization (99.7th percentile)
3. **Resizing**: Images resized to 448×448 for inference
4. **ImageNet Normalization**: Standard normalization for pretrained features

## Utilities

- **`checkdirctexist()`**: Directory creation utility
- **`normalize()`**: Image normalization for model input
- **`bias_field_correction()`**: N4 bias field correction using SimpleITK
- **`convert_to_cbf()`**: Segmentation mask visualization

## File Structure
```
├── model.py # Model architectures
├── inference.py # Core inference functions
├── inference_hemi.py # Hemisphere segmentation script
├── inference_complete.py # Complete segmentation script
├── utils.py # Utility functions
└── requirements.txt # Dependencies
```

## Dependencies

Key dependencies include:
- PyTorch
- torchvision
- nibabel
- pydicom
- dicom2nifti
- SimpleITK
- opencv-python
- numpy
- matplotlib

## Notes

- The models expect GPU availability (CUDA)
- DICOM files are automatically converted to NIfTI format if needed
- Multi-planar DICOM series are filtered to keep only axial slices
- Edge slices (first/last 2 slices) are excluded from segmentation

## Citation

If you use this code in your research, please cite the relevant paper associated with this implementation.

## License

Please refer to the license file in the repository for usage terms.