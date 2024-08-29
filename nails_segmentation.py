# nails_segmentation.py

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch.optim import Adam
from torch import nn

# 定义图像路径和标签路径
img_path = "D:\nails-segmentation/images/"
lbl_path = "D:\nails-segmentation/labels/"

# 获取所有图像的文件名列表
l = [i for i in os.listdir(img_path)]

# 自定义数据集类，用于加载图像和对应的标签
class CustomDst(Dataset):
    def __init__(self, list_of_img, image_path, labels_path, transform=None):
        self.image_path = image_path
        self.labels_path = labels_path
        self.transform = transform
        self.list_of_img = list_of_img
    
    def __len__(self):
        # 返回数据集的大小
        return len(self.list_of_img)
    
    def __getitem__(self, idx):
        # 加载图像和对应的标签
        img = Image.open(self.image_path + self.list_of_img[idx])
        label = Image.open(self.labels_path + self.list_of_img[idx])
        
        # 对图像进行标准化处理
        img = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(ToTensor()(img))
        
        # 如果有额外的transform操作，应用它们
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
            # 处理标签，使其成为二值化的掩码
            label = label[1]
            label = torch.where(label > 0, 1.0, 0.0).unsqueeze(0)
            return img, label
        else:
            return img, label

# 显示一个示例图像及其标签
def show_sample_image():
    img = Image.open(img_path + "09aefeec-e05f-11e8-87a6-0242ac1c0002.jpg")
    img = Resize((224, 224))(Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(ToTensor()(img)))
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    
    img = Image.open(lbl_path + "09aefeec-e05f-11e8-87a6-0242ac1c0002.jpg")
    img = Resize((224, 224))(ToTensor()(img))
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

# 创建数据集并分割成训练集和测试集
dataset = CustomDst(l, img_path, lbl_path, Resize((224, 224)))
print("Dataset length:", len(dataset))
train, test = torch.utils.data.random_split(dataset, (45, 7))

# 创建数据加载器
train_loader = DataLoader(train, batch_size=2, shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=2)

# 检查数据集中的一个样本
def check_sample():
    for i in dataset:
        plt.imshow(i[0].permute(1, 2, 0))
        plt.show()
        print(i[1].shape)
        plt.imshow(i[1][0])
        plt.show()
        break

# 定义和初始化DeepLabV3模型
def initialize_model():
    model = models.segmentation.deeplabv3_resnet101(weights=True)
    model.classifier[4] = nn.Conv2d(256, 1, 1)  # 修改分类器使其适应单通道输出
    model.aux_classifier[4] = nn.Conv2d(256, 1, 1)  # 修改辅助分类器
    return model

model = initialize_model()
print(f"{sum(p.numel() for p in model.parameters()):,} total parameters.")
print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} training parameters.")

# 定义优化器和损失函数
opt = Adam(model.parameters(), lr=1e-5)
loss_fn = nn.BCEWithLogitsLoss()

# 训练模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

for epoch in range(20):
    running_loss = 0.0
    items = 0.0
    model.train()
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        yhat = model(X)
        loss = loss_fn(yhat['out'], y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        running_loss += loss
        items += y.size(0)
    
    print(f"Epoch {epoch} train_running_loss {running_loss/items}")
    
    # 验证模型
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        items = 0.0
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            yhat = model(X)
            loss = loss_fn(yhat['out'], y)
            running_loss += loss
            items += y.size(0)
        print(f"         test_running_loss {running_loss/items}")

# 在测试集上进行预测并显示结果
def predict_and_show():
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            n = 0
            X, y = X[n].to(device), y[n].to(device)

            predMask = model(X.unsqueeze(0))
            predMask = torch.sigmoid(predMask['out'])
            predMask = predMask.cpu().numpy()
            predMask = (predMask > 0.5) * 255

            plt.imshow(X.permute(1, 2, 0).cpu())
            plt.show()
            plt.imshow(predMask[0][0])
            plt.show()
            plt.imshow(y[0].cpu())
            plt.show()

if __name__ == "__main__":
    show_sample_image()
    check_sample()
    predict_and_show()

