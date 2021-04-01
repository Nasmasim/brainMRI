# Age Regression from Brain MRI
The objective of this project is to implement two supervised learning approaches for age regression from brain MRI. Predicting the age of a patient from their brain MRI scan can have diagnostic value for a number of diseases that may cause structural changes and potential damage to the brain. A discrepancy between the predicted age and the real, chronological age of a patient might indicate the presence of disease. This requires an accurate predictor of brain age which may be learned from a set of healthy reference subjects, given their brain MRI data and their actual age.

# Imaging Data 
[BrainMRI.ipynb](https://github.com/Nasmasim/brainMRI/blob/main/BrainMRI.ipynb) walks through different pre-processing steps using ```SimpleITK```. 
<p align="center">
<img src="https://github.com/Nasmasim/brainMRI/blob/main/images/MRI_images.png" width="50%">
</p>

After applying pre-processing (intensity normalization) and downsampling, we aim to regress the age of a subject using the volumes of brain tissues as features. The brain tissues include grey matter (GM), white matter (WM), and cerebrospinal fluid (CSF).

<p align="center">
<img src=https://github.com/Nasmasim/brainMRI/blob/main/images/MRI_images_preprocessed.png width="50%">
</p>


