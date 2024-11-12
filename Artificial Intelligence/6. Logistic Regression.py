# Logistic Regression을 구현해봅시다.

로지스틱 회귀는 Regression이라는 이름과 달리, Classification을 수행합니다. 
그렇다면 어째서 Regression인가 ?

Logit(Log-Odds, 승산)을 Linear Regression 함으로써 분류가 가능하기 때문입니다.
Logit은 1-q/q로 계산되며, 이를 Sigmoid Function에 통과시키면 0~1 사이의 확률값이 도출됩니다.
그 확률값을 Threshold 기준으로 Binary Classification하면 됩니다.

조금 더 자세히 설명하자면, 1 / 1+Exp^Weight*x+Bias 를 계산함으로써 도출되는 0~1 사이의 값이 Threshold보다 크면 1, 작으면 0으로 분류하는 것입니다.

Loss는 Binary Cross Entropy Loss를 사용하며, 이 손실함수를 최적화하는 방향으로 Gradient Descent를 수행합니다.

- 송현교 작성

-----------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as f

torch.manual_seed(1) # 동일한 결과 출력을 위해 시드 설정

x_train = torch.FloatTensor([[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]])
y_train = torch.FloatTensor([[0],[0],[0],[1],[1],[1]]) # 이전까지의 라벨 값과 다르게, Binary한 값을 라벨로 줍니다.

Weight = torch.zeros((2,1), requires_grad=True)
Bias = torch.zeros(1, requires_grad = True)

Optimizer = torch.optim.SGD((Weight, Bias), lr = 0.1)

epochs = 1000
for epoch in range(epochs):
    Hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(Weight)) + Bias)) # Logistic Regression의 경우, Weight와 Bias에 대한 Sigmoid Function이 Hypothesis(H(x))입니다.
    # Hypothesis = torch.sigmoid(x_train.matmul(Weight) + Bias)처럼, Pytorch에서 제공하는 Sigmoid Function을 호출하는 것이 더 간단하긴 합니다.
    Loss = torch.mean(-(y_train * torch.log(Hypothesis) + (1 - y_train) * torch.log(1 - Hypothesis))) # Binary Classification Model이므로, Binary Cross Entropy Loss를 손실함수로 사용합니다.
    # Loss = f.binary_cross_entropy(Hypothesis, y_train)와 같이, Pytorch에서 제공하는 이진 크로스 엔트로피 로스 함수 호출이 더 간편합니다.
    
    prediction = Hypothesis > 0.5 # 이진 분류이므로, 0.5를 Threshold로 하여 분류합니다.

    Optimizer.zero_grad()
    Loss.backward()
    Optimizer.step()

    print("Epoch {}/{} Hypothesis {} Loss {}".format(epoch,epochs,prediction,Loss))



