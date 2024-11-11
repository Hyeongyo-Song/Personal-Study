#Linear Regression + Gradient Descent

import torch

x_train = torch.FloatTensor([[0],[1],[2]])
y_train = torch.FloatTensor([[0],[1],[2]])

weight = torch.zeros(1, True)
bias = torch.zeros(1, True)

epochs = 10
for epoch in range(epochs):
    Hypothesis = weight * x_train + bias
    loss = torch.mean((Hypothesis - y_train) ** 2)
    Gradient = 2 * torch.mean((Hypothesis - y_train) * x_train)
    lr = 0.1
    weight -= lr * Gradient

    print('Epoch {}/{} W:{}, B:{}, Loss:{}'.format(epoch, epochs, weight, bias, loss))
