import os
import pandas as pd
import torch
import torchvision
from riceDataset import riceDataset


def load_train_data(base_path):
    label_dict={'Arborio':0,'basmati':1,'Ipsala':2,'Jasmine':3,'Karacadag':4}
    cate=list(label_dict.keys())
    files=[]
    for key in cate:
        filenames = os.walk(os.path.join(base_path,key))
        files+=list(filenames)[0][2]
    files=[file for file in files if file.endswith(".jpg")]
    labels=[label_dict[file.split(' ')[0]] for file in files]
    file_df=pd.DataFrame({"file":files,"label":labels})
    file_df["file"]=file_df.file.map(lambda x: os.path.join(base_path,x.split(' ')[0],x))
    return file_df

Train_dataset = riceDataset(load_train_data('./Rice_Image_Dataset/Train/'),transform=torchvision.transforms.ToTensor())
print(Train_dataset.__getitem__(5)) 


