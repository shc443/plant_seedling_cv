
import torch
from torch import optim
from transformers import AdamW, AutoModelForMaskedLM, AutoConfig
from transformers import get_linear_schedule_with_warmup

import timm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.functional import accuracy, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import seaborn as sns
import matplotlib.pyplot as plt
import re
import os 
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer

from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from torchmetrics import Accuracy

import pytorch_lightning as pl
# from torchvision import transforms
# import torchvision.transforms as tt
from torchvision import transforms

from torchvision.datasets import ImageFolder
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse

import numpy as np
import pandas as pd
import random
import time
import datetime
import pickle 
from PIL import Image
from tqdm import tqdm
tqdm.pandas()

from os import listdir
from os.path import isfile, join

from model import PlantClassifier
from data_plant import PlantDataset, TestPlant, PlantDataModule, find_classes

class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
def main(args):
    root_dir = os.getcwd() + "/data/train"
    # root_dir2 = os.getcwd() + "/plant/data/test"
    classes, class_to_idx, idx_to_class, df = find_classes(root_dir)
    num_classes = len(classes)
    
    df2 = pd.concat([df]*5, ignore_index=True)
    df_train, df_val = train_test_split(df2,
                                        test_size=0.1, 
                                        random_state=1230, 
                                        stratify=df2['class_index'])
    
    val_tfms = transforms.Compose([
                #transforms.CenterCrop(356),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    test_dataset = TestPlant(rootdir = os.getcwd() + "/data/test", transform=val_tfms)
    test_loader = DataLoader(test_dataset, batch_size = 32, num_workers= 4, shuffle=False)
    
    df_test = pd.DataFrame(os.listdir(os.getcwd() + "/data/test"))
    df_test[1] = df_test.shape[0]*[0]
    
    # checkpoint_callback = pl.callbacks.ModelCheckpoint()
    tmpdir = os.getcwd() + '/model_weights/'
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_last=True, monitor="val_loss",filename='sample-{epoch:02d}-{val_loss:.2f}')
    
    # Initialize a trainer
    if wandb:
        wandb.login()
        wandb.init(project="stat_final", entity="kchoi1230")
        wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')
        trainer = pl.Trainer(max_epochs = args.epochs,
                             progress_bar_refresh_rate=1, 
                             gpus=1, 
                             checkpoint_callback=[checkpoint_callback])
    else:
        trainer = pl.Trainer(max_epochs = args.epochs,
                             progress_bar_refresh_rate=1, 
                             gpus=1, 
                             logger=wandb_logger,
                             checkpoint_callback=[checkpoint_callback])
    
    model=PlantClassifier(model_name = args.model)
    pdm = PlantDataModule(train_df = df2, val_df = df2.iloc[:10], test_df = df2.iloc[:10], batch_size=32, data_dir = root_dir)
    pdm.setup()
    
    trainer.fit(model, pdm)
    trainer.save_checkpoint("./plant/model_weights/last.ckpt")
    
    test_result = trainer.predict(dataloaders=test_loader)
    test_result_list = np.hstack(pd.Series(test_result).apply(lambda x : x.tolist()))
    df_test[1] = pd.Series(test_result_list).apply(lambda x : idx_to_class[x])
    df_test.columns = ['file','species']
    df_test.to_csv('test_sample4.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Deep Learning model")
    # parser.add_argument("--save_path", type=str, default=os.getcwd()+"/results", help="Path to save the results")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--wandb", type=str, default=True, help="Login wandb T/F")
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="Choose model from timm pkg")
    args = parser.parse_args()
    main(args)