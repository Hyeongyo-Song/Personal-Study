# Multivariable Linear Regression을 nn.Module로 편하게 구현하시오.

nn.Module 이라는 Abstract Class를 상속받으면 Linear Regression 뿐만 아니라 추후 서술할 여러 모델들을 편리하게 구현할 수 있답니다.

- 송현교 작성

-------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn # 다변수 선형회귀 구현을 편하게 만들어주는 nn.Module을 불러오기 위해 Import하였습니다.
import torch.nn.functional as F # 본 문서에서는 사전에 정의된 Loss 함수를 불러오기 위해 Import하였습니다.

class MultivariableLinearRegression(nn.Module): # MultivariableLinearRegression 클래스는 nn.Module를 상속받습니다.
    def __init__(self): # Initialization
        super().__init__()
        self.linear = nn.Linear(3,1) # nn 모듈에 포함된 선형회귀 함수입니다. 단순히 입력 텐서의 개수와 출력 텐서의 개수를 파라미터로 주기만 하면 됩니다.

    def forward(self,x): # __forward__가 아닌 그냥 forward입니다. 이름 틀리면 작동 안됩니다.
        return self.linear(x)

model = MultivariableLinearRegression()

x_train = torch.FloatTensor([[73,80, 75],
                             [93,88,93],
                             [89,91,90],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5) # Stochastic Gradient Descent 사용합니다. 아까 만든 다변수 선형회귀 클래스.parameters()와 학습률만 주면 됩니다.

Epochs = 20 # 총 20회 반복
for epoch in range(Epochs):
    Hypothesis = y_train 
    predict = model(x_train) # 이전에 비해, 코드가 훨씬 간결해졌습니다.

    loss = F.mse_loss(predict, Hypothesis) # 손실함수도 이미 정의된 함수를 불러오기만 하면 됩니다.

    optimizer.zero_grad() # 이 3개는 그냥 외워야 합니다. 세트입니다.
    loss.backward() # 세트 !
    optimizer.step() # 세트 !

    print("epoch {}/{} predict {} Loss {}".format(epoch,Epochs,predict,loss))

