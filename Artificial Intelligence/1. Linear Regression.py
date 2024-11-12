# Linear Regression을 구현합시다.

Linear Regression은 독립변수 x를 통해 종속변수 y를 예측하는 것이며,
예측을 위해 x 간의 규칙을 가장 잘 표현하는 하나의 Linear한 직선을 찾는 것이 바로 Linear Regression(선형 회귀)입니다.

주로 사용하는 Loss로는 Mean Absolute Error와 Mean Squared Error가 있으며,
MAE는 |Ypredict - Ytrue|의 평균, MSE는 (Ypredict - Ytrue)^2의 평균을 의미합니다.
Outlier가 적당히 무시되길 바란다면 MAE, Outlier를 고려해야 할 땐 MSE를 사용합니다.

- 송현교 작성

---------------------------------------------------------------------------------

import torch

x_train = torch.FloatTensor([[0],[1],[2]])
y_train = torch.FloatTensor([[0],[1],[2]])

weight = torch.zeros(1, requires_grad=True) # weight를 1로 초기화, gradient 업데이트 할 것이므로 True
bias = torch.zeros(1, requires_grad=True) # bias도 1로 초기화, gradient 업데이트 할 것이므로 True

optimizer = torch.optim.SGD([weight,bias], lr = 0.01) # Optimizer를 Stochastic Gradient Descent로 사용하겠음, Learning Rate(나아가는 정도)는 0.01.

nb_epochs = 1000 # 총 Epoch수.
for epoch in range(1, nb_epochs+1):
    hypothesis = weight * x_train + bias # 예측값 Y Predict를 Hypothesis라 정의, 이는 Wx + b로 도출할 수 있습니다.
    cost = torch.mean((hypothesis - y_train) ** 2) # Mean Squared Error(MSE)를 Loss Function으로 사용합니다. 본 문서에서는 직접 구현.

    optimizer.zero_grad() # 이전에 구한 Gradient가 있으면, 0으로 초기화함. 초기화하지 않으면 기존의 Gradient에 계속 더해짐.
    cost.backward() # Loss를 BackPropagation 합니다.
    optimizer.step() # Loss를 최소화하는 방향으로 학습을 진행 !

print(weight * 3 + bias) # 3이라는 실제 값을 주었을때 Predict 값은 ?
