
import tensorflow as tf
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from datasets.augmentation_cifar import resize_with_crop
from datasets.augmentation_cifar import resize_with_crop_aug

from config import config
conf = config.flags
from tonic.datasets import CIFAR10DVS
import tonic
from module import DataGenerator
def load():
   sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
   transforms = tonic.transforms.Compose(
       [tonic.transforms.ToImage(
           sensor_size=sensor_size,
       ),]
   )
   dataset = tonic.datasets.CIFAR10DVS(save_to="./data", transform=transforms)
   train_size = int(0.8*len(dataset))
   valid_size = len(dataset) - train_size
   train, valid = random_split(dataset, [train_size,valid_size])
   train_ds = DataGenerator(DataLoader(train, shuffle=True, batch_size=conf.batch_size),10)
   valid_ds = DataGenerator(DataLoader(valid, shuffle=True, batch_size=conf.batch_size),10)
   train_ds_num = train_size
   valid_ds_num = valid_size
   return train_ds, valid_ds, valid_ds, train_ds_num, valid_ds_num, valid_ds_num
