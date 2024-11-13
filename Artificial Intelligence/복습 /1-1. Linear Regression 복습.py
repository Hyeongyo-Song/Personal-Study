# Linear Regression을 외부 도움, 자료 참고 없이 스스로 구현해봅시다. x_train과 y_train만 기존의 것을 사용합니다.

import torch
import torch.nn.functional as f

x_train = torch.FloatTensor([[0],[1],[2]])
y_train = torch.FloatTensor([[0],[1],[2]])

Weight = torch.zeros(1, requires_grad = True)
Bias = torch.zeros(1, requires_grad = True)

Optimizer = torch.optim.SGD([Weight, Bias], lr = 0.1)

Epochs = 20
for epoch in range(Epochs):
    Hypothesis = Weight * x_train + Bias
    Loss = f.mse_loss(Hypothesis, y_train)

    Optimizer.zero_grad()
    Loss.backward()
    Optimizer.step()

    print("Epoch {}/{} Hypothesis {} Loss {}".format(epoch,Epochs,Hypothesis,Loss))

