# Copyright NYU KevinCHOI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re
import pytorch_lightning as pl
import pickle
from torchvision import transforms
from collections import Counter
from PIL import Image
from torchvision.datasets import ImageFolder

class PlantDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.transform = transform
        self.df = dataframe
        self.root_dir = root_dir
        #self.classes = 
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        fullpath = os.path.join(self.root_dir, self.df.iloc[idx][0])
        image = Image.open(fullpath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.df.iloc[idx][2]
    
class TestPlant(Dataset):
    def __init__(self, rootdir, transform=None):
        self.transform=transform
        self.rootdir=rootdir
        self.image_files = os.listdir(self.rootdir)
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        fullpath = os.path.join(self.rootdir, self.image_files[idx])
        image = Image.open(fullpath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

class PlantDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, batch_size, data_dir: str = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.num_classes = 12
        self.validation_split_ratio = 0.1
  
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        
        self.train_tfms = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
            #transforms.CenterCrop(356),
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_tfms = transforms.Compose([
            #transforms.CenterCrop(356),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # self.train_tfms = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)), 
        #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # self.val_tfms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)),
        #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.train_dataset = PlantDataset(self.train_df , self.data_dir, self.train_tfms)
        self.val_dataset = PlantDataset(self.val_df, self.data_dir, self.val_tfms)
        self.test_dataset = PlantDataset(self.test_df , self.data_dir, self.val_tfms)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
def find_classes(fulldir):
    classes = os.listdir(fulldir)
    classes.sort()
    class_to_idx = dict(zip(classes, range(len(classes)))) 
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    train = []
    
    for i, label in idx_to_class.items():
        path = fulldir+"/"+label
        for file in os.listdir(path):
            train.append([f'{label}/{file}', label, i])
    df = pd.DataFrame(train, columns=["file", "class", "class_index"]) 
    return classes, class_to_idx, idx_to_class, df

