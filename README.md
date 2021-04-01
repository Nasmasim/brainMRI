# Age Regression from Brain MRI
The objective of this project is to implement two supervised learning approaches for age regression from brain MRI. Predicting the age of a patient from their brain MRI scan can have diagnostic value for a number of diseases that may cause structural changes and potential damage to the brain. A discrepancy between the predicted age and the real, chronological age of a patient might indicate the presence of disease. This requires an accurate predictor of brain age which may be learned from a set of healthy reference subjects, given their brain MRI data and their actual age.

# Imaging Data 
[BrainMRI.ipynb](https://github.com/Nasmasim/brainMRI/blob/main/BrainMRI.ipynb) walks through different pre-processing steps using ```SimpleITK```. After applying pre-processing (intensity normalization) and downsampling in [data_helpers.py](https://github.com/Nasmasim/brainMRI/blob/main/utils/data_helpers.py), we aim to regress the age of a subject using the volumes of brain tissues as features. The brain tissues include grey matter (GM), white matter (WM), and cerebrospinal fluid (CSF).

<p align="center">
<img src=https://github.com/Nasmasim/brainMRI/blob/main/images/MRI_images_preprocessed.png width="50%">
</p>

# 3D ResUnet
We implement a 3D ResUnet architecture in [UNet3d.py](https://github.com/Nasmasim/brainMRI/blob/main/UNet3d.py) based on the paper by Özgün Çiçek et al. ['3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation'](https://arxiv.org/abs/1606.06650) onsisting of a contracting encoder that analyses the whole image and a subsequent decoder which produces high-resolution segmentation. Each convolution block in the encoder contains 3x3x3 convolutions, each followed by a ReLu and then a 2x2x2 max pooling with strides of two in each dimension. We added residuals to the output as a convolution of the input of each block, thus implementing a ResUNet. 
## Segmentation Results 

<p align="center">
<img src=https://github.com/Nasmasim/brainMRI/blob/main/images/3dUnet_result.png width="50%">
</p>


