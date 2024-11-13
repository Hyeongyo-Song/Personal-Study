# Softmax Classification을 구현해봅시다.

import torch

x_train = [[1,2,1,1],
           [2,1,3,2],
           [3,1,3,4],
           [4,1,5,5],
           [1,7,5,5],
           [1,2,5,6],
           [1,6,6,6],
           [1,7,7,7]]
y_train = [2,2,2,1,1,1,0,0] # x_train의 각 행이 어느 클래스에 속하는지를 나타내는 Label임.

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train) # 분류기의 출력은 Long 타입이 일반적임.

Weight = torch.zeros((4,3), requires_grad = True)
Bias = torch.zeros(1, requires_grad = True)

Optimizer = torch.optim.SGD([Weight, Bias], lr = 0.1)

Epochs = 1000
for epoch in range(Epochs):
    Hypothesis = torch.nn.functional.softmax(x_train.matmul(Weight) + Bias, dim=1) # 행을 기준으로 Softmax Function 적용.

    y_one_hot = torch.zeros_like(Hypothesis) # Hypothesis와 같은 크기의 Matrix를 생성합니다.
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1) # y_train을 열벡터로 펼치고, 그 벡터가 가지는 값을 인덱스로 하여 x_train에 1을 뿌립니다.

    Loss = torch.mean(-1 * (y_one_hot * torch.log(Hypothesis))) # 손실함수는 Cross Entropy Loss를 사용합니다. -Ytrue log(Ypredict)로 계산합니다.
    # Loss = (y_one_hot * -torch.log(torch.nn.functional.softmax(Hypothesis, dim=1))).sum(dim=1).mean()도 같은 기능을 수행합니다.

    Optimizer.zero_grad()
    Loss.backward()
    Optimizer.step()

    print("Epoch {}/{} Loss {}".format(epoch,Epochs,Loss))