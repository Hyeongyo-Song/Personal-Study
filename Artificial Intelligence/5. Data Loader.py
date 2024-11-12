# DataLoader를 상속받음으로써 대량의 데이터를 불러와 처리하는 방법을 숙지합시다.

- 송현교 작성

-------------------------------------------------------------------------------


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class MultiVariableLinearRegression(nn.Module): # 이전 문서에서 익힌 것처럼, nn.Module를 상속받아 편하게 다변수 선형회귀 모델을 생성합시다. 이전에 다뤘으므로 자세한 설명은 생략합니다.
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)

    def forward(self, x):
        return self.linear(x)


class CustomDataset(Dataset): # 여태까지는 데이터의 양이 적어서 수월했찌만, 앞으로는 대량의 데이터를 사용해야 합니다. 이번 문서에서는 DataLoader와 Dataset을 Import하여 그 실력을 키워 보겠습니다.
    def __init__(self): # 중요한 것은 Dataset을 불러와 사용하는 것이므로, 이번에는 제가 임의로 데이터셋을 정의했습니다.
        self.x_data = [[73,85,75],
                       [93,88,93],
                       [89,91,90],
                       [96,98,100],
                       [73,66,70]]
        self.y_data = [[152],[185],[180],[196],[142]]

    def __len__(self): # 데이터의 개수를 return하는 메서드입니다.
        return len(self.x_data)

    def __getitem__(self, idx): # 특정 Index의 데이터를 return합니다.
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x,y

Dataset = CustomDataset()

dataLoader = DataLoader(
    Dataset, # 데이터셋은 아까 생성한 Custom Dataset을 사용해 줍시다.
    batch_size = 2, # 배치 크기는 모델이 한 에포크를 처리할 때, 몇 개의 샘플 데이터로 분리하여 처리할 지 결정해주는 HyperParameter입니다. 2의 배수로 설정해 주는 것이 메모리 운용에 유리합니다.
    shuffle = True # 한 배치 내에 데이터를 샘플링할때 데이터들을 랜덤하게 섞어서 샘플링할 지 결정합니다. 일반적으로 True로 설정해 준다고 배웠습니다.
)

model = MultiVariableLinearRegression()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) # 언제나와 같이 Stochastic Gradient Descent를 사용합니다.

Epochs = 20 # 총 20회 반복
for epoch in range(Epochs):
    for batch_index, samples in enumerate(dataLoader): # Enumerate를 사용한 이유는 사용자에게 배치 처리 진행 과정을 출력해주기 위해서입니다. 따라서 range로 진행해도 상관은 없습니다.
        x_train, y_train = samples # dataLoader 내의 데이터를 차례차례 samples에 넣습니다. dataLoader의 값들은 데이터셋과 라벨 값으로 구성되었으므로 이를 분리하여 x_train과 y_train에 삽입합니다.

        prediction = model(x_train) # 데이터셋에 대한 Predict 값을 아주 편하게 구할 수 있게 되었습니다.
        loss = F.mse_loss(prediction, y_train) # 이제 손실함수를 직접 구현할 줄 알기 때문에, Mean Sqaured Error를 불러와 편하게 사용합시다.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epochs {}/{} Batch {}/{} prediction {} Loss {}'.format(epoch, Epochs, batch_index, len(dataLoader), prediction, loss))
