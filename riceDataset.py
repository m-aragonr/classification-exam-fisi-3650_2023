# Imports
import os
import pandas as pd
import torch
from skimage import io

# Class containing a rice dataset
class riceDataset:

    # Constructor method
    def __init__(self,file_df,transform=None):
        self.file_df = file_df
        self.transform = transform
    
    # Returns number of images in the dataset
    def __len__(self):
        return len(self.file_df)
    
    # Returns a tuple containing the image as an RGB array and its label
    # index: index of the image to retrieve
    def __getitem__(self,index):
        image = io.imread(self.file_df[index,0])
        t_label = torch.tensor(int(self.file_df.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image,y_label)