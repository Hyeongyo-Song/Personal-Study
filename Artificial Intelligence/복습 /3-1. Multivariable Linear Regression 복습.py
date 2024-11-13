# 다차원 데이터에 대한 Linear Regression을 외부 도움, 자료 참조 없이 스스로 구현해봅시다. 데이터만 기존의 것을 사용합니다.

import torch

x_train = torch.FloatTensor([[73,80, 75],
                             [93,88,93],
                             [89,91,90],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

Weight = torch.zeros((3,1), requires_grad = True)
Bias = torch.zeros(1, requires_grad = True)

Epochs = 1000
for epoch in range(Epochs):
    Hypothesis = x_train.matmul(Weight) + Bias
    Loss = torch.mean((Hypothesis - y_train) ** 2)

    Gradient_Descent = 2 * torch.mean(Hypothesis - y_train)
    Learning_Rate = 1e-5

    Weight = Weight - Gradient_Descent * Learning_Rate

    print("Epochs {}/{} Hypothesis {} Loss {}".format(epoch,Epochs,Hypothesis,Loss))

# 구현하는 과정에서 Epoch의 중요성을 알게 되었음. Epoch를 20으로 설정하니 Loss가 거의 줄어들지 않는 것을 확인. 적절한 Epoch를 찾는 것도 중요하네요.
