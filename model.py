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

import pytorch_lightning as pl
import torch
from torch import optim
import torch.nn.functional as F
from torchmetrics.functional import auroc
from transformers import AdamW#, AutoModelForMaskedLM, AutoConfig
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import pickle
import timm
from torchvision import transforms
from torchmetrics import Accuracy

class PlantClassifier(pl.LightningModule):
    """
    Multi-class Classification model for Plant seedling Detection Task
    """
    def __init__(self, 
                 lr=1e-4, 
                 n_classes=12, 
                 model_name = None):
        """
        Args:
            :param n_classes : Number of classes to classify
            :param model_name: Name of the model i.e. "maxxvit_rmlp_small_rw_256"
        """
        super().__init__()
        self.lr=lr
        self.save_hyperparameters()
        # self.model= timm.create_model('vit_base_patch16_224_in21k',pretrained=True,num_classes=n_classes)
        # self.model= timm.create_model('maxvit_base_tf_512',pretrained=True, num_classes=0)
        
        self.model= timm.create_model(model_name, pretrained=True, num_classes=n_classes)
        self.accuracy = Accuracy()
        
    def configure_optimizers(self):
        opt= optim.AdamW(self.model.parameters(),self.lr)
        return opt
    
    def training_step(self, dl, idx):
        x,y=dl
        z = self.model(x)
        loss = F.cross_entropy(z,y)
        
        preds = torch.argmax(z, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, dl, idx):
        x,y = dl
        z = self.model(x)
        loss = F.cross_entropy(z,y)
        
        preds = torch.argmax(z, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def test_step(self, dl, idx):
        x,y = dl
        z = self.model(x)
        preds = torch.argmax(z, dim=1)
        # self.log("test_loss", loss, prog_bar=True, logger=True)
        return preds

    def predict_step(self, dl, idx, dataset_idx=0):
        x,y = dl
        z = self.model(x)
        preds = torch.argmax(z, dim=1)
        # self.log("predic_loss", loss, prog_bar=True, logger=True)
        return preds
