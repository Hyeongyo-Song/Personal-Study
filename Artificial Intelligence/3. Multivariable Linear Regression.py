# 3. Multivariate Linear Regression을 구현하시오.

import torch

x_train = torch.FloatTensor([[73,80, 75],
                             [93,88,93],
                             [89,91,90],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

weight = torch.zeros((3,1), requires_grad = True) # 입력 Tensor가 3개, 출력 Tensor는 1개임을 의미합니다. 표로 나타내면 쉽게 알 수 있습니다.
bias = torch.zeros(1, requires_grad = True) # 1로 초기화합니다.

optimizer = torch.optim.SGD([weight, bias], 1e-5) # Optimizer는 언제나처럼 Stochastic Gradient Descent를 사용합니다. 두번째 파라미터는 'lr = 1e-5'의 형식이 되어야 옳습니다.

epochs = 1000 # 총 1000회 반복합니다.
for epoch in range(epochs):
    Hypothesis = x_train.matmul(weight) + bias # Y = Wx + b를 코드로 표현합니다.
    cost = torch.mean((Hypothesis - y_train) ** 2) # Mean Squared Error 사용.

    optimizer.zero_grad() # 언제나 세트로 와야 합니다.
    cost.backward() # 세트 !
    optimizer.step() # 세트 !

    print('epoch:{}/{} Hypothesis:{} Loss:{}'.format(epoch,epochs,Hypothesis,cost))
