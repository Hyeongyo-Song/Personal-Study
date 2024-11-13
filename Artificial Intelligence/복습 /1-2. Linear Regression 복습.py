# 조금 더 High Level로 Linear Regression을 구현해봅시다. nn.Module을 사용합니다.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x_train):
        return self.linear(x_train)

Model = Linear_Regression()

x_train = torch.FloatTensor([[0],[1],[2],[3],[4],[5],[6]])
y_train = torch.FloatTensor([[0],[1],[2],[3],[4],[5],[6]])

Optimizer = torch.optim.SGD(Model.parameters(), lr = 1e-5)

Epochs = 20
for epoch in range(Epochs):
    Hypothesis = Model(x_train)
    Loss = F.mse_loss(Hypothesis, y_train)

    Optimizer.zero_grad()
    Loss.backward()
    Optimizer.step()

    print("Epoch {}/{} Hypothesis {} Loss {}".format(epoch,Epochs,Hypothesis,Loss))

# 다차원 데이터에 대한 Linear Regression을 공부할 때 사용한 코드와 다를 게 거의 없는데, Hypothesis와 Loss가 매우 불규칙적임. 원인 분석이 필요.
