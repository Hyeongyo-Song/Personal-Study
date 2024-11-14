# Softmax Classification 모델을 스스로 구현해봅시다.

import torch

x_train = [[1,2,1,1],
           [2,1,3,2],
           [3,1,3,4],
           [4,1,5,5],
           [1,7,5,5],
           [1,2,5,6],
           [1,6,6,6],
           [1,7,7,7]]
y_train = [2,2,2,1,1,1,0,0]

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

Weight = torch.zeros((4,3), requires_grad = True)
Bias = torch.zeros(1, requires_grad = True)

Optimizer = torch.optim.SGD([Weight, Bias], lr = 0.1)

Epochs = 1000
for epoch in range(Epochs):
    Hypothesis = torch.nn.functional.softmax(x_train.matmul(Weight) + Bias)
    y_one_hot = torch.zeros_like(Hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

    Loss = torch.mean(-1 * (y_one_hot * torch.log(Hypothesis)))

    Optimizer.zero_grad()
    Loss.backward()
    Optimizer.step()

    print("{}/{}, {}, {}".format(epoch,Epochs,Hypothesis,Loss))
