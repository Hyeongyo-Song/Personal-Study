# High Level의 Multivariable Linear Regression을 구현해봅시다. nn.Module을 사용합니다.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Multivariable_Linear_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    def forward(self, x):
        return self.linear(x)

x_train = torch.FloatTensor([[73,80, 75],
                             [93,88,93],
                             [89,91,90],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

Model = Multivariable_Linear_Regression()
Optimizer = torch.optim.SGD(Model.parameters(), lr = 1e-5)

Epochs = 20
for epoch in range(Epochs):
    Hypothesis = Model(x_train)
    Loss = F.mse_loss(Hypothesis, y_train)

    Optimizer.zero_grad()
    Loss.backward()
    Optimizer.step()

    print("Epochs {}/{} Hypothesis {} Loss {}".format(epoch,Epochs,Hypothesis,Loss))