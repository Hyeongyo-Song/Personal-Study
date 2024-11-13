# 이번에는 Linear Regression을 혼자만의 힘으로 구현하되, Gradient Descent와 Loss도 직접 구현합시다.

import torch

x_train = torch.FloatTensor([[0],[1],[2]])
y_train = torch.FloatTensor([[0],[1],[2]])

Weight = torch.zeros(1, requires_grad = True)
Bias = torch.zeros(1, requires_grad = True)

Epochs = 20
for epoch in range(Epochs):
    Hypothesis = Weight * x_train + Bias
    Loss = torch.mean((Hypothesis - y_train) ** 2) # Mean Squared Error, (Ypredict - Ytrue)^2

    Gradient_Descent = 2 * torch.mean((Hypothesis - y_train)) # Loss를 미분함으로써 Gradient를 구할 수 있음.
    Learning_Rate = 0.1

    Weight = Weight - Gradient_Descent * Learning_Rate

    print("Epoch {}/{} Hypothesis {} Loss {}".format(epoch,Epochs,Hypothesis,Loss))