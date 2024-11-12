import torch
import torch.nn.functional as f

torch.manual_seed(1) # 동일한 결과 출력을 위해 시드 설정

x_train = torch.FloatTensor([[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]])
y_train = torch.FloatTensor([[0],[0],[0],[1],[1],[1]])

Weight = torch.zeros((2,1), requires_grad=True)
Bias = torch.zeros(1, requires_grad = True)

Optimizer = torch.optim.SGD((Weight, Bias), lr = 0.1)

epochs = 1000
for epoch in range(epochs):
    Hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(Weight)) + Bias))
    # Hypothesis = torch.sigmoid(x_train.matmul(Weight) + Bias)
    Loss = torch.mean(-(y_train * torch.log(Hypothesis) + (1 - y_train) * torch.log(1 - Hypothesis)))

    prediction = Hypothesis > 0.5

    Optimizer.zero_grad()
    Loss.backward()
    Optimizer.step()

    print("Epoch {}/{} Hypothesis {} Loss {}".format(epoch,epochs,prediction,Loss))



