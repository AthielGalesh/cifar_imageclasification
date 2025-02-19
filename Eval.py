import numpy as np, pandas as pd
from matplotlib.pyplot import subplots
from sklearn.linear_model import (LinearRegression, LogisticRegression, Lasso)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from sklearn.model_selection import (train_test_split, GridSearchCV)
from matplotlib import pyplot as plt

#torch libraries------
import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset

from torchmetrics import (MeanAbsoluteError, R2Score)
from torchinfo import summary
from torchvision.io import read_image

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from lightning_lite.utilities.seed import seed_everything

#import pytorch_lightning_lite.utilities.seed.seed_everything

seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)

#torchvision------
from torchvision.datasets import MNIST,CIFAR100
from torchvision.models import (resnet50,
                                ResNet50_Weights)
from torchvision.transforms import (Resize,
                                    Normalize,
                                    CenterCrop,
                                    ToTensor)

from ISLP.torch import (SimpleDataModule,
                        SimpleModule,
                        ErrorTracker,
                        rec_num_workers)

from ISLP.torch.imdb import (load_lookup,
                             load_tensor,
                             load_sparse,
                             load_sequential)
#import glob and json for finding matches, needed to build our own images
from glob import glob
import json
#from lab_10_CIFAR import CIFARModel, BuildingBlock

#-----Clase de CIFAR---------------------------------------
#----------------------------------------------------------
class BuildingBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(BuildingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3,3),
                              padding="same")
        self.activation=nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self,x):
        return self.pool(self.activation(self.conv(x)))

class CIFARModel(nn.Module):

    def __init__(self):
        super(CIFARModel, self).__init__()
        sizes=[(3,32),
               (32,64),
               (64,128),
               (128,256)]
        self.conv = nn.Sequential(*[BuildingBlock(in_,out_)
                                    for in_, out_ in sizes])
        self.output=nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(2*2*256,512),
                                  nn.ReLU(),
                                  nn.Linear(512,100))

    def forward(self,x):
        val = self.conv(x)
        val = torch.flatten(val, start_dim=1)
        return self.output(val)
#------------------------------------------------------------------------
#------------------------------------------------------------------------

pretrained_cifar = CIFARModel()

pretrained_cifar.load_state_dict(torch.load("cifar_cnn.pth"))
pretrained_cifar.eval()
print("cargado con exito")


max_num_workers = rec_num_workers()
resize = Resize((232,232))
crop = CenterCrop(224)
normalize = Normalize([0.485,0.456,0.406],
                      [0.229,0.224,0.225])
imgfiles = sorted([f for f in glob('book_images/*')])
imgs = torch.stack([torch.div(crop(resize(read_image(f))), 255)
                    for f in imgfiles])
imgs = normalize(imgs)
#imgs.size()

resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
summary(resnet_model,
        input_data=imgs,
        col_names=["input_size",
                   "output_size",
                   "num_params"])
resnet_model.eval()

img_preds = resnet_model(imgs)
img_probs=np.exp(np.asarray(img_preds.detach()))
img_probs /= img_probs.sum(1)[:,None]

labs = json.load(open("imagenet_class_index.json"))
class_labels = pd.DataFrame([(int(k),v[1]) for k, v in labs.items()],
                            columns=["idx","label"])
class_labels=class_labels.set_index("idx")
class_labels=class_labels.sort_index()

for i, imgfile in enumerate(imgfiles):
    img_df = class_labels.copy()
    img_df["prob"]=img_probs[i]
    img_df=img_df.sort_values(by="prob", ascending=False)[:3]
    print(f"Image: {imgfile}")
    print(img_df.reset_index().columndrop["idx"])
