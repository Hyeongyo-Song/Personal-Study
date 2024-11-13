# 대량의 데이터를 불러오기 위한 연습으로, DataLoader를 활용하여 Regression하는 모델을 직접 제작해봅시다.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Custom_Datasets(Dataset):
    def __init__(self):
        super().__init__()
        self.x_data = [[73,85,75],
                       [93,88,93],
                       [89,91,90],
                       [96,98,100],
                       [73,66,70]]
        self.y_data = [[152],[185],[180],[196],[142]]
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y

class Multivariable_Linear_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    def forward(self, x):
        return self.linear(x)

dataSet = Custom_Datasets()
dataLoader = DataLoader(dataSet, batch_size=2, shuffle=True)

Model = Multivariable_Linear_Regression()
Optimizer = torch.optim.SGD(Model.parameters(), lr = 1e-5)

Epochs = 1000
for epoch in range(Epochs):
    for index, data in enumerate(dataLoader):
        x_train, y_train = data
        Hypothesis = Model(x_train)
        Loss = F.mse_loss(Hypothesis, y_train)

        Optimizer.zero_grad()
        Loss.backward()
        Optimizer.step()

        print("Epochs {}/{} Hypothesis {} Loss {}".format(epoch,Epochs,Hypothesis,Loss))


