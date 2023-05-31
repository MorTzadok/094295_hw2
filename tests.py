import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as T
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm
from PIL import Image

#count images in dolder 
base_dir = os.path.join("..", "data copy")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
count = 0

path = train_dir

for number in os.listdir(path):
    if number !='.DS_Store':
        number_count = len([i for i in os.listdir(path+'/'+ str(number))])
        print("number of images in " + str(number) + " is : "+  str(number_count))
        count+= number_count
         
print(count)

