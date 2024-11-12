# 1. Linear Regression과 Mean Squared Error를 구현하시오.

import torch

x_train = torch.FloatTensor([[0],[1],[2]])
y_train = torch.FloatTensor([[0],[1],[2]])

weight = torch.zeros(1, requires_grad=True) # weight를 1로 초기화, gradient 업데이트 할 것이므로 True
bias = torch.zeros(1, requires_grad=True) # bias도 1로 초기화, gradient 업데이트 할 것이므로 True

optimizer = torch.optim.SGD([weight,bias], lr = 0.01) # Optimizer를 Stochastic Gradient Descent로 사용하겠음, Learning Rate(나아가는 정도)는 0.01.

nb_epochs = 1000 # 총 Epoch(몇번 반복?)수.
for epoch in range(1, nb_epochs+1):
    hypothesis = weight * x_train + bias # 예측값 Y Predict를 Hypothesis라 정의, 이는 Wx + b로 도출할 수 있습니다.
    cost = torch.mean((hypothesis - y_train) ** 2) # Mean Squared Error(MSE)를 Loss Function으로 사용합니다. 본 문서에서는 직접 구현.

    optimizer.zero_grad() # 이전에 구한 Gradient가 있으면, 0으로 초기화함. 초기화하지 않으면 기존의 Gradient에 계속 더해짐.
    cost.backward() # Loss를 BackPropagation 합니다.
    optimizer.step() # Loss를 최소화하는 방향으로 학습을 진행 !

print(weight * 3 + bias) # 3이라는 실제 값을 주었을때 Predict 값은 ?
