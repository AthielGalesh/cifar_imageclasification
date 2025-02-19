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
max_num_workers = rec_num_workers()


if __name__ == "__main__":
    (cifar_train,
     cifar_test) = [CIFAR100(root="data",
                             train=train,
                             download=True,
                             transform=ToTensor())
                    for train in [True, False]]


# cifar_train_X = torch.stack([transform(x) for x in cifar_train.data])
# cifar_test_X = torch.stack([transform(x) for x in cifar_test.data])
# cifar_train = TensorDataset(cifar_train_X,
#                             torch.tensor(cifar_train.targets))
# cifar_test = TensorDataset(cifar_test_X,
#                            torch.tensor(cifar_test.targets))

    cifar_train
    cifar_dm=SimpleDataModule(cifar_train,
                              cifar_test,
                              validation=0.2,
                              num_workers=max_num_workers,
                              batch_size=128)

    for idx, (X_,Y_) in enumerate(cifar_dm.train_dataloader()):
        print("X: ", X_.shape)
        print("Y: ", Y_.shape)
        if idx >=1:
            break

    fig, axes = subplots(5,5,figsize=(10,10))
    rng = np.random.default_rng(3)
    indices = rng.choice(np.arange(len(cifar_train)),25,
                         replace=False).reshape((5,5))

    for i in range(5):
        for j in range(5):
            idx=indices[i,j]
            axes[i,j].imshow(np.transpose(cifar_train[idx][0],
                                          [1,2,0]),
                             interpolation=None)
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
    #plt.show()

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

    cifar_model=CIFARModel()
    summary(cifar_model,
            input_data=X_,
            col_names=["input_size",
                       "output_size",
                       "num_params"])


    cifar_optimizer=RMSprop(cifar_model.parameters(), lr=0.001)
    cifar_module=SimpleModule.classification(cifar_model, num_classes=100,
                                             optimizer=cifar_optimizer
                                             )
    cifar_logger=CSVLogger("logs", name="CIFAR100")

    cifar_trainer = Trainer(deterministic=True,
                            max_epochs=30,
                            logger=cifar_logger,
                            callbacks=[ErrorTracker()])
    cifar_trainer.fit(cifar_module,
                      datamodule=cifar_dm)

    cifar_result=cifar_trainer.test(cifar_module,
                       datamodule=cifar_dm)

    #guardo la rn entrenada:----------
    try:
        torch.save(cifar_model.state_dict(), "cifar_cnn.pth")
        print("modelo guardado con exito")
    except:
        raise ValueError("Error")
#-------------------------------------
#-------------------------------------
#-------------------------------------

