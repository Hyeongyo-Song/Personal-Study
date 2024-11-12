import torch
import torch.nn as nn
import torch.nn.functional as F

class MultivariableLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)

    def forward(self,x):
        return self.linear(x)

model = MultivariableLinearRegression()

x_train = torch.FloatTensor([[73,80, 75],
                             [93,88,93],
                             [89,91,90],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)

Epochs = 20
for epoch in range(Epochs):
    Hypothesis = y_train
    predict = model(x_train)

    loss = F.mse_loss(predict, Hypothesis)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("epoch {}/{} predict {} Loss {}".format(epoch,Epochs,predict,loss))

