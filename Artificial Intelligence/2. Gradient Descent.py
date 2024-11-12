# Gradient Descent를 직접 구현하시오.

Gradient Descent는 우리말로 경사하강법이라 하며, Loss 즉 손실함수를 최적화하는데 사용하는 최적화 기법입니다.
Loss Function이 Multivariable Function이라 가정했을 때, 각 변수들을 Weight로 편미분하면 Gradient를 구할 수 있습니다.

이 Gradient는 수학적으로 그래프의 가장 가파른 지점을 향합니다.
그렇다면 ? Gradient가 향하는 방향의 반대로 나아가면 극솟값에 도달할 수 있습니다 !
이것이 바로 Gradient Descent입니다.

Learning Rate를 곱해줌으로써 나아가는 정도도 조절해 줄 수 있답니다.

------------------------------------------------------------------

import torch

x_train = torch.FloatTensor([[0],[1],[2]])
y_train = torch.FloatTensor([[0],[1],[2]])

weight = torch.zeros(1, True) # 두번째 Parameter는 'requires_grad = True'가 되어야 합니다. 본 문서에서는 우연히 오류가 없었지만 다른 상황에서는 문제가 발생함을 확인하였음.
bias = torch.zeros(1, True) # 마찬가지입니다.

epochs = 10 # 총 10번 반복할 예정
for epoch in range(epochs):
    Hypothesis = weight * x_train + bias # 정답 값 Y = Wx + b (W는 Weight, b는 bias를 의미합니다.)
    loss = torch.mean((Hypothesis - y_train) ** 2) # 이전 문서와 동일하게 Mean Squared Error를 사용합니다. 잔차제곱평균이라고도 합니다.
    Gradient = 2 * torch.mean((Hypothesis - y_train) * x_train) # 이전 문서에서는 optim.SGD를 통해 편하게 확률적 경사 하강법을 적용했지만, 본 문서에서는 직접 구현합니다. Gradient는 MSE를 Weight로 편미분하면 얻을 수 있습니다.
    lr = 0.1 # 학습률(Learning Rate)입니다.
    weight -= lr * Gradient # Gradient가 음수라면 우측으로, 양수라면 좌측으로 움직입니다. Learning Rate를 곱해줌으로써 이를 가능케 합니다.

    print('Epoch {}/{} W:{}, B:{}, Loss:{}'.format(epoch, epochs, weight, bias, loss))
