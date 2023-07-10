import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from collections import OrderedDict



class Model():

    def __init__(self):
        self.model = self.load_model("rice_model.pth")
        return
    
    def load_model(self,file_path):
    
        load = torch.load(file_path)
        
        model = models.vgg16(pretrained=True)
            
        for param in model.parameters():
            param.requires_grad = False
        
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                                ('relu', nn.ReLU()),
                                                ('drop', nn.Dropout(p=0.5)),
                                                ('fc2', nn.Linear(5000, 102)),
                                                ('output', nn.LogSoftmax(dim=1))]))

        model.classifier = classifier

        model.load_state_dict(load['model_state_dict'])
        
        return model
    

    def process_image(self,image_path):
    
        pil_image = Image.open(image_path)
        
        # Resize
        if pil_image.size[0] > pil_image.size[1]:
            pil_image.thumbnail((5000, 256))
        else:
            pil_image.thumbnail((256, 5000))
            
        # Crop 
        left_margin = (pil_image.width-224)/2
        bottom_margin = (pil_image.height-224)/2
        right_margin = left_margin + 224
        top_margin = bottom_margin + 224
        
        pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
        
        # Normalize
        np_image = np.array(pil_image)/255
        mean = np.array([0.1158,0.1162,0.1204])
        std = np.array([0.2915,0.2927,0.3023])
        np_image = (np_image - mean) / std
    
        np_image = np_image.transpose((2, 0, 1))
        
        return np_image

    def predict(self,file_path):

        image = self.process_image(file_path)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = image.unsqueeze(0)
        
        output = self.model.forward(image)
        
        probabilities = torch.exp(output)
        top_probabilities, top_indices = probabilities.topk(1)
        top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 

        return top_indices[0]
