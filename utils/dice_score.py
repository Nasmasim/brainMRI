import numpy as np
import torch

def dice_score(prd,seg, num_classes):
    dice = np.zeros(num_classes)
    for tissue in range(num_classes):
        #For each tissue, count the pixels that match between truth and prediction and multiply it by 2
        #Divide it by the total number of pixels predicted as this tissue type plus the actual number of pixels 
        #of this tissue type according to the true segmentation
        dice[tissue] = (torch.sum(prd[seg.squeeze(1)==tissue]==tissue))*2.0 / ( torch.sum(prd[prd==tissue]==tissue) + torch.sum(seg[seg==tissue]==tissue) )
    return dice