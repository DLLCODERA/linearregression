import gzip
import os
import glob
import cv2
import device
import numpy
import numpy as np
import torch
import torchvision
from torch import nn
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, dataloader
from skimage import io, transform

from torch.autograd import Variable
import torch.nn.functional as F

# # 训练数据和测试数据的下载
# 训练数据和测试数据的下载
trainDataset = torchvision.datasets.MNIST( # torchvision可以实现数据集的训练集和测试集的下载
    root="./data", # 下载数据，并且存放在data文件夹中
    train=True, # train用于指定在数据集下载完成后需要载入哪部分数据，如果设置为True，则说明载入的是该数据集的训练集部分；如果设置为False，则说明载入的是该数据集的测试集部分。
    transform=torchvision.transforms.ToTensor(), # 数据的标准化等操作都在transforms中，此处是转换
    download=True # 瞎子啊过程中如果中断，或者下载完成之后再次运行，则会出现报错
)

testDataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath: # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, folder, data_name, label_name,transform=None):
        (train_set, train_labels) = load_data(folder, data_name, label_name)  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)
# 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
trainDataset = DealDataset('data/MNIST/raw/', "train-i")