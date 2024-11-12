# 3. Multivariate Linear Regression을 구현하시오.

import torch

x_train = torch.FloatTensor([[73,80, 75],
                             [93,88,93],
                             [89,91,90],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

weight = torch.zeros((3,1), requires_grad = True)
bias = torch.zeros(1, requires_grad = True)

optimizer = torch.optim.SGD([weight, bias], 1e-5)

epochs = 1000
for epoch in range(epochs):
    Hypothesis = x_train.matmul(weight) + bias
    cost = torch.mean((Hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('epoch:{}/{} Hypothesis:{} Loss:{}'.format(epoch,epochs,Hypothesis,cost))
