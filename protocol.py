import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import Conv2d_cd, CDCNpp
import time
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

train_image_dir = '/home/fas1/Meison/Train_images/'
test_image_dir = '/home/fas1/Meison/Test_images/'
val_image_dir = '/home/fas1/Meison/val_images/'


model = CDCNpp().cuda()
model.load_state_dict(torch.load('CDCNpp_P4_290.pkl'))
model.eval()

count = 0
sum = 0
torch.cuda.empty_cache()
for videoname in os.listdir(train_image_dir):
    video_path = os.path.join(train_image_dir, videoname)
    if videoname[7] == '1':
        print(videoname)
        for img in os.listdir(video_path):
            image = os.path.join(video_path, img)
            image = cv2.imread(image)

            f = image[:, :, ::-1].transpose((2, 0, 1))
            f = f[np.newaxis, :, :, :]
            f = np.array(f)
            f = torch.from_numpy(f.astype(np.float)).float()
            # normalization
            f = (f - 127.5) / 128
            f = f.cuda()
            map_x = model(f)
            tmp = torch.sum(map_x)
            sum += tmp
            count += 1



        print('average = ', (sum/count))

