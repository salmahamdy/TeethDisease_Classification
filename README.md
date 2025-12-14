# Dental Disease Classification

Deep learning models for automated classification of dental conditions from X-ray images using CNN and VGG16 architectures.

## Dataset Structure

```
Teeth_Dataset/
├── Training/
├── Validation/
└── Testing/
```

The dataset contains 7 classes of dental conditions organized into training, validation, and test sets.

## Models

### 1. Custom CNN 

Built from scratch with 5 convolutional blocks:
- Progressive filter increase: 32 → 64 → 128 → 128 → 64
- Max pooling after each conv layer
- 50% dropout for regularization
- Dense layers: 128 units + 7-class output

**Training config:**
- Optimizer: Adam (lr=0.001)
- Loss: Sparse categorical crossentropy
- Epochs: 20
- Batch size: 32

Includes horizontal flip augmentation during training.

### 2. VGG16 Transfer Learning 

Pre-trained VGG16 (ImageNet weights) with frozen base layers:
- Global average pooling
- Two dense layers (256 units each)
- 50% dropout
- 7-class softmax output

**Training config:**
- Optimizer: Adam (lr=0.001)
- Epochs: 100
- Batch size: 32

## Setup

```bash
pip install tensorflow matplotlib numpy scikit-learn
```

## Usage

Update dataset paths in the scripts to match your directory structure:
```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    "YOUR_PATH/Training",
    image_size=(224, 224),
    batch_size=32
)
```

Run either model:
```bash
python DentalImageClassification.py
python Dental_Image_Classification_VGG.py
```

## Evaluation

Both scripts output:
- Training/validation accuracy and loss curves
- Test set accuracy
- Confusion matrix (VGG16 script)
- Classification report with precision/recall/F1
- Multi-class ROC curves

## Model Saving

- CNN: Saved as `cnn_model.h5`
- VGG16: Saved as `Dental_VGG16_finetuned.h5`

## Requirements

- TensorFlow 2.x
- Python 3.7+
- 224x224 RGB images
- GPU recommended for training


All images are normalized to [0,1] range. The VGG16 model typically achieves better performance due to pre-trained features but takes longer to train with 100 epochs.
```
