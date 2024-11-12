# Multivariate Linear Regression을 구현하시오.

이전에 구현해본 Linear Regression은 단순한 Y = Wx+b 형태의 구조였습니다.
이를 Simple Linear Regression이라 합니다.

그렇다면 x와 W가 여러 개일 때는 어떻게 해야 할까요 ?

본 문서에서는 Input Tensor와 Output Tensor의 크기를 맞춰주고, matmul Method를 사용하여 다차원의 데이터를 쉽게 처리할 수 있었습니다. 함께 보실까요 ?

- 송현교 작성

-------------------------------------------------------------------------------------------------------------------------------

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
    Hypothesis = x_train.matmul(weight) + bias # Y = W1x1 + W2x2 + W3x3 + ... + Wnxn + b를 코드로 표현합니다.
    cost = torch.mean((Hypothesis - y_train) ** 2) # Mean Squared Error 사용.

    optimizer.zero_grad() # 언제나 세트로 와야 합니다.
    cost.backward() # 세트 !
    optimizer.step() # 세트 !

    print('epoch:{}/{} Hypothesis:{} Loss:{}'.format(epoch,epochs,Hypothesis,cost))
