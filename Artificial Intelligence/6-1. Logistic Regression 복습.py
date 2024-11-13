# 로지스틱 회귀를 외부 자료 참고 없이 직접 구현해보자 ! 데이터는 기존의 것을 사용.

import torch
import torch.nn.functional as F
import torch.nn as nn

x_train = torch.FloatTensor([[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]])
y_train = torch.FloatTensor([[0],[0],[0],[1],[1],[1]])

Weight = torch.zeros((2,1), requires_grad = True)
Bias = torch.zeros(1, requires_grad = True)

Optimizer = torch.optim.SGD([Weight, Bias], lr = 0.1)

Epochs = 1000
for epoch in range(Epochs):
    Hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(Weight)) + Bias))
    Loss = -1 * (torch.mean(y_train * torch.log(Hypothesis) + (1 - y_train) * torch.log(1 - Hypothesis)))

    Optimizer.zero_grad()
    Loss.backward()
    Optimizer.step()

    print("Epochs {}/{} Hypotheis {} Loss {}".format(epoch,Epochs,Hypothesis,Loss))