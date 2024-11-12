# 1. Linear Regression과 Mean Squared Error를 구현하시오.

import torch

x_train = torch.FloatTensor([[0],[1],[2]])
y_train = torch.FloatTensor([[0],[1],[2]])

weight = torch.zeros(1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([weight,bias], lr = 0.01)

nb_epochs = 1000
for epoch in range(1, nb_epochs+1):
    hypothesis = weight * x_train + bias
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print(weight * 3 + bias)
